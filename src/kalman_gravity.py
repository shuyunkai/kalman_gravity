"""核心算法：卡尔曼滤波重力场重建 + Parker频域水深反演 + 挠曲均衡约束。"""
import numpy as np
from scipy.ndimage import gaussian_filter
import warnings

from utils import sediment_gravity_correction as _sediment_gravity_correction

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*overflow.*")


class KalmanGravityReconstructor:
    """标准卡尔曼滤波：融合卫星(1')和船载(15'')重力数据，重建高分辨率重力场。

    状态变量 x: 高分辨率格网 (ny*upscale, nx*upscale)
    观测1 z_sat: 低分辨率卫星重力 (ny, nx)，观测模型为块均值
    观测2 z_ship: 稀疏船载重力，观测模型为点采样
    状态转移: 空间高斯平滑，模拟重力场连续性先验
    """

    def __init__(self, grid_shape, region_size_deg, lr_grid_shape=None):
        self.grid_shape = grid_shape
        self.ny, self.nx = grid_shape
        self.n_state = self.ny * self.nx
        self.region_size_deg = region_size_deg

        if lr_grid_shape is None:
            self.lr_ny = max(1, int(self.ny / 4))
            self.lr_nx = max(1, int(self.nx / 4))
        else:
            self.lr_ny, self.lr_nx = lr_grid_shape

        self.x = None
        self._initialized = False

    def initialize(self, gravity_lr, gravity_ship=None,
                   process_noise_std=1.0, sat_noise_std=2.0, ship_noise_std=0.5,
                   ship_train_mask=None):
        """初始化滤波器：上采样卫星数据作为初值，设定噪声参数。

        ship_train_mask: 船载训练/验证拆分掩码 (True=用于约束, False=保留验证).
                         避免船载数据既做初始状态又被重新观测的循环。
        """
        from scipy.ndimage import zoom
        zoom_y = self.ny / gravity_lr.shape[0]
        zoom_x = self.nx / gravity_lr.shape[1]
        x0 = zoom(gravity_lr, (zoom_y, zoom_x), order=1)

        if gravity_ship is not None:
            if ship_train_mask is not None:
                train = ship_train_mask & ~np.isnan(gravity_ship)
                x0[train] = gravity_ship[train]
            else:
                valid = ~np.isnan(gravity_ship)
                x0[valid] = gravity_ship[valid]

        self.x = x0.ravel()
        init_var = (process_noise_std ** 2) * np.ones(self.n_state)
        self.P_diag = init_var.copy()
        self.Q_diag = (process_noise_std ** 2) * np.ones(self.n_state)
        self.R_sat_val = sat_noise_std ** 2
        self.R_ship_val = ship_noise_std ** 2
        self.gravity_lr = gravity_lr
        self.gravity_ship = gravity_ship
        self.ship_train_mask = ship_train_mask
        self._initialized = True
        self._iter_count = 0

        print(f"卡尔曼滤波器初始化完成. 状态维度: {self.n_state}")
        print(f"  过程噪声 std: {process_noise_std:.2f} mGal")
        print(f"  卫星观测噪声 std: {sat_noise_std:.2f} mGal")
        print(f"  船载观测噪声 std: {ship_noise_std:.2f} mGal")

    def predict(self, smooth_sigma=1.0):
        """预测步：空间高斯平滑 + 协方差膨胀（模拟过程噪声）。"""
        if not self._initialized:
            raise RuntimeError("请先调用 initialize() 初始化滤波器")

        x_2d = self.x.reshape(self.ny, self.nx)
        x_pred_2d = gaussian_filter(x_2d, sigma=smooth_sigma)
        self.x = x_pred_2d.ravel()

        factor = 1.0 + smooth_sigma * 0.1
        self.P_diag = self.P_diag * factor + self.Q_diag
        positive = self.P_diag[self.P_diag > 0]
        cap = 100.0 * (np.median(positive) + 1e-10) if positive.size > 0 else 1e10
        self.P_diag = np.minimum(self.P_diag, cap)

    def update_satellite(self):
        """更新步-卫星观测：用低分辨率格网约束高分辨率重建结果。

        每个低分辨率像元对应一块高分辨率像元，约束块均值接近卫星观测值。
        NOTE: 使用块均值 P 和标量近似 Kalman gain，导致块内 P 均一化。
        这是高维状态下的计算效率近似，块内各格点的空间不确定性信息会丢失。
        """
        if not self._initialized:
            raise RuntimeError("请先调用 initialize() 初始化滤波器")

        x_2d = self.x.reshape(self.ny, self.nx)
        lr_ny, lr_nx = self.gravity_lr.shape
        hr_per_lr_y = self.ny // lr_ny
        hr_per_lr_x = self.nx // lr_nx

        for i in range(lr_ny):
            for j in range(lr_nx):
                i0 = i * hr_per_lr_y
                i1 = min((i + 1) * hr_per_lr_y, self.ny)
                j0 = j * hr_per_lr_x
                j1 = min((j + 1) * hr_per_lr_x, self.nx)

                block = x_2d[i0:i1, j0:j1]
                block_flat = block.ravel()
                z = self.gravity_lr[i, j]

                z_pred = block_flat.mean()
                P_block = self.P_diag.reshape(self.ny, self.nx)[i0:i1, j0:j1].mean()
                K = P_block / (P_block + self.R_sat_val)
                innovation = z - z_pred
                block_flat_updated = block_flat + K * innovation
                x_2d[i0:i1, j0:j1] = block_flat_updated.reshape(block.shape)
                P_updated = (1 - K) * P_block
                self.P_diag.reshape(self.ny, self.nx)[i0:i1, j0:j1] = P_updated

        self.x = x_2d.ravel()

    def update_shipborne(self, max_points_per_update=5000, prop_radius=3, prop_fraction=0.3):
        """更新步-船载观测：逐点卡尔曼更新 + 高斯邻域传播。

        船载重力在每个测点上高精度观测，更新后通过邻域传播补偿对角协方差
        忽略的空间相关性。
        """
        if not self._initialized:
            raise RuntimeError("请先调用 initialize() 初始化滤波器")
        if self.gravity_ship is None:
            return

        valid_mask = ~np.isnan(self.gravity_ship)
        if self.ship_train_mask is not None:
            valid_mask = valid_mask & self.ship_train_mask
        valid_indices = np.where(valid_mask.ravel())[0]
        if len(valid_indices) == 0:
            print("  无有效船载观测点, 跳过更新")
            return

        n_obs = min(len(valid_indices), max_points_per_update)
        if len(valid_indices) > max_points_per_update:
            seed = 42 + self._iter_count
            rng = np.random.default_rng(seed)
            chosen = rng.choice(len(valid_indices), n_obs, replace=False)
            obs_indices = valid_indices[chosen]
        else:
            obs_indices = valid_indices
        self._iter_count += 1

        x_2d = self.x.reshape(self.ny, self.nx)
        P_2d = self.P_diag.reshape(self.ny, self.nx)
        ship_flat = self.gravity_ship.ravel()

        correction_field = np.zeros_like(x_2d)
        weight_field = np.zeros_like(x_2d)

        for idx in obs_indices:
            i, j = divmod(idx, self.nx)
            z = ship_flat[idx]
            z_pred = x_2d[i, j]
            P_ii = P_2d[i, j]
            K = P_ii / (P_ii + self.R_ship_val)
            innovation = z - z_pred
            x_2d[i, j] += K * innovation
            P_2d[i, j] = (1 - K) * P_ii

            i0 = max(0, i - prop_radius)
            i1 = min(self.ny, i + prop_radius + 1)
            j0 = max(0, j - prop_radius)
            j1 = min(self.nx, j + prop_radius + 1)

            sy, sx = i1 - i0, j1 - j0
            dy = np.abs(np.arange(sy) - (i - i0)).reshape(sy, 1)
            dx = np.abs(np.arange(sx) - (j - j0)).reshape(1, sx)
            dist = np.sqrt(dx * dx + dy * dy)
            w = np.exp(-0.5 * (dist / (prop_radius / 2.0)) ** 2)
            w = w / w.sum()

            correction_field[i0:i1, j0:j1] += K * innovation * w
            weight_field[i0:i1, j0:j1] += w

        has_weight = weight_field > 0
        no_ship = ~valid_mask
        mask_prop = has_weight & no_ship
        if mask_prop.sum() > 0:
            blend = correction_field[mask_prop] / np.maximum(weight_field[mask_prop], 1e-10)
            x_2d[mask_prop] += prop_fraction * blend
            info_gain = prop_fraction * np.minimum(weight_field[mask_prop], 1.0)
            P_2d[mask_prop] = P_2d[mask_prop] * np.maximum(1.0 - info_gain, 0.1)

        self.x = x_2d.ravel()
        self.P_diag = P_2d.ravel()

    def smooth_backward(self, x_history, x_pred_history, P_diag_history, smooth_sigma=0.8):
        """RTS 反向平滑：利用前向滤波历史状态和协方差，获得更优估计。"""
        factor = 1.0 + smooth_sigma * 0.1
        n_steps = len(x_history)
        x_smooth = x_history[-1].copy()
        P_smooth = P_diag_history[-1].copy()

        for k in range(n_steps - 2, -1, -1):
            P_pred = P_diag_history[k] * factor + self.Q_diag
            C_k = P_diag_history[k] / np.maximum(P_pred, 1e-10)
            x_smooth = x_history[k] + C_k * (x_smooth - x_pred_history[k + 1])
            P_smooth = P_diag_history[k] + C_k ** 2 * (P_smooth - P_pred)
            P_smooth = np.maximum(P_smooth, 0)

        return x_smooth, P_smooth

    def run(self, n_iterations=5, smooth_sigma=0.8, verbose=True):
        """运行迭代预测-更新循环 + RTS 反向平滑，返回重建的高分辨率重力场。"""
        if not self._initialized:
            raise RuntimeError("请先调用 initialize() 初始化滤波器")

        self._iter_count = 0
        x_history = []
        x_pred_history = []
        p_history = []

        for it in range(n_iterations):
            self.predict(smooth_sigma=smooth_sigma)
            x_pred_history.append(self.x.copy())
            self.update_satellite()
            if self.gravity_ship is not None:
                self.update_shipborne()

            x_history.append(self.x.copy())
            p_history.append(self.P_diag.copy())

            if verbose:
                x_2d = self.x.reshape(self.ny, self.nx)
                print(f"  迭代 {it+1}/{n_iterations}: mean={x_2d.mean():.2f}, std={x_2d.std():.2f}, "
                      f"range=[{x_2d.min():.1f}, {x_2d.max():.1f}]")

        x_final, P_smooth = self.smooth_backward(x_history, x_pred_history, p_history, smooth_sigma=smooth_sigma)
        self._P_final = P_smooth.reshape(self.ny, self.nx)
        return x_final.reshape(self.ny, self.nx)


class EnsembleKalmanGravityReconstructor:
    """集合卡尔曼滤波 (EnKF)：Monte Carlo 逼近全协方差，避免显式构建大矩阵。

    注意: update_satellite() 对每个 LR 像素分配 n_state 大小的数组，
    内存消耗为 O(n_state * n_lr)，>100×100 格网请用标准 KF。
    """

    def __init__(self, grid_shape, region_size_deg, n_ensemble=50, seed=42):
        self.grid_shape = grid_shape
        self.ny, self.nx = grid_shape
        self.n_state = self.ny * self.nx
        self.region_size_deg = region_size_deg
        self.n_ensemble = n_ensemble
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.ensemble = None
        self._initialized = False

    def initialize(self, gravity_lr, gravity_ship=None,
                   process_noise_std=1.0, sat_noise_std=2.0, ship_noise_std=0.5,
                   ship_train_mask=None):
        """初始化 EnKF 集合：以卫星数据上采样为均值，加随机扰动。"""
        from scipy.ndimage import zoom
        zoom_y = self.ny / gravity_lr.shape[0]
        zoom_x = self.nx / gravity_lr.shape[1]
        x0 = zoom(gravity_lr, (zoom_y, zoom_x), order=1).ravel()

        if gravity_ship is not None:
            if ship_train_mask is not None:
                train = ship_train_mask & ~np.isnan(gravity_ship)
                x0_2d = x0.reshape(self.ny, self.nx)
                x0_2d[train] = gravity_ship[train]
                x0 = x0_2d.ravel()
            else:
                valid = ~np.isnan(gravity_ship)
                x0_2d = x0.reshape(self.ny, self.nx)
                x0_2d[valid] = gravity_ship[valid]
                x0 = x0_2d.ravel()

        self.ship_train_mask = ship_train_mask

        # Memory check: ensemble is (n_ensemble, n_state) float64
        mem_gb = self.n_ensemble * self.n_state * 8 / (1024**3)
        if mem_gb > 8:
            raise MemoryError(
                f"EnKF ensemble requires {mem_gb:.1f} GB memory "
                f"({self.n_ensemble}x{self.n_state}). Use standard KF instead.")

        ensemble_std = process_noise_std * 2
        self.ensemble = self.rng.standard_normal(
            (self.n_ensemble, self.n_state)) * ensemble_std
        self.ensemble += x0[None, :]

        self.gravity_lr = gravity_lr
        self.gravity_ship = gravity_ship
        self.sat_noise_std = sat_noise_std
        self.ship_noise_std = ship_noise_std
        self.process_noise_std = process_noise_std
        self._initialized = True

        print(f"EnKF 初始化完成. 集合大小: {self.n_ensemble}, 状态维度: {self.n_state}")

    def predict(self, smooth_sigma=1.0):
        """EnKF 预测步：对每个集合成员做空间平滑并加过程噪声。"""
        for m in range(self.n_ensemble):
            x_2d = self.ensemble[m].reshape(self.ny, self.nx)
            x_smooth = gaussian_filter(x_2d, sigma=smooth_sigma)
            self.ensemble[m] = x_smooth.ravel()
            self.ensemble[m] += self.rng.standard_normal(self.n_state) * self.process_noise_std * 0.3

    def update_satellite(self):
        """EnKF 更新-卫星观测：逐低分辨率像元标量观测更新每个集合成员。"""
        lr_ny, lr_nx = self.gravity_lr.shape
        hr_per_lr_y = self.ny // lr_ny
        hr_per_lr_x = self.nx // lr_nx

        for i in range(lr_ny):
            for j in range(lr_nx):
                i0 = i * hr_per_lr_y
                i1 = min((i + 1) * hr_per_lr_y, self.ny)
                j0 = j * hr_per_lr_x
                j1 = min((j + 1) * hr_per_lr_x, self.nx)

                block_mask = np.zeros(self.n_state, dtype=bool)
                for row in range(i0, i1):
                    block_mask[row * self.nx + j0: row * self.nx + j1] = True
                n_block = block_mask.sum()
                if n_block == 0:
                    continue

                y_ens = (self.ensemble[:, block_mask].mean(axis=1)
                         + self.rng.standard_normal(self.n_ensemble) * self.sat_noise_std)
                self._enkf_update_scalar(y_ens, self.gravity_lr[i, j], self.sat_noise_std ** 2)

    def update_shipborne(self):
        """EnKF 更新-船载观测：逐有效船载点标量观测更新。"""
        if self.gravity_ship is None:
            return

        valid_mask = ~np.isnan(self.gravity_ship)
        if self.ship_train_mask is not None:
            valid_mask = valid_mask & self.ship_train_mask
        valid_flat = self.gravity_ship.ravel()
        valid_indices = np.where(valid_mask.ravel())[0]
        n_obs = min(len(valid_indices), 500)
        if len(valid_indices) > 500:
            seed = 42 + getattr(self, '_enkf_iter_count', 0)
            rng = np.random.default_rng(seed)
            chosen = rng.choice(len(valid_indices), n_obs, replace=False)
            valid_indices = valid_indices[chosen]
        self._enkf_iter_count = getattr(self, '_enkf_iter_count', 0) + 1

        for idx in valid_indices:
            z = valid_flat[idx]
            y_ens = (self.ensemble[:, idx]
                     + self.rng.standard_normal(self.n_ensemble) * self.ship_noise_std)
            self._enkf_update_scalar(y_ens, z, self.ship_noise_std ** 2)

    def _enkf_update_scalar(self, y_ens, z, R):
        """单点标量观测的 EnKF 更新（随机扰动法）。"""
        x_mean = self.ensemble.mean(axis=0)
        y_mean = y_ens.mean()
        X_prime = self.ensemble - x_mean[None, :]
        Y_prime = y_ens - y_mean

        cov_xy = (X_prime * Y_prime[:, None]).sum(axis=0) / (self.n_ensemble - 1)
        var_y = Y_prime.var(ddof=1)
        K = cov_xy / np.maximum(var_y + R, 1e-10)

        for m in range(self.n_ensemble):
            innovation = z - y_ens[m]
            self.ensemble[m] += K * innovation

    def run(self, n_iterations=5, smooth_sigma=0.8, verbose=True):
        """运行 EnKF 迭代，返回集合均值作为高分辨率重力场。"""
        if not self._initialized:
            raise RuntimeError("请先调用 initialize() 初始化 EnKF")

        self._enkf_iter_count = 0

        for it in range(n_iterations):
            self.predict(smooth_sigma=smooth_sigma)
            self.update_satellite()
            if self.gravity_ship is not None:
                self.update_shipborne()

            x_mean = self.ensemble.mean(axis=0).reshape(self.ny, self.nx)
            if verbose:
                print(f"  迭代 {it+1}/{n_iterations}: mean={x_mean.mean():.2f}, std={x_mean.std():.2f}")

        return self.ensemble.mean(axis=0).reshape(self.ny, self.nx)


class KalmanBathymetryInverter:
    """卡尔曼滤波水深反演：从重建的重力异常通过 Parker 频域公式反演海底地形。

    观测模型: F[g](k) = 2πGΔρ·e^{-kd}·F[b](k)，含 Tikhonov 正则化和
    岩石圈挠曲均衡校正。
    """

    def __init__(self, grid_shape, region_size_deg, lat_center=0):
        self.grid_shape = grid_shape
        self.ny, self.nx = grid_shape
        self.n_state = self.ny * self.nx
        self.region_size_deg = region_size_deg
        self.dx_deg = region_size_deg / self.nx
        self.lat_center = lat_center

        self.G = 6.674e-11
        self.rho_c = 2670
        self.rho_w = 1030
        self.drho = self.rho_c - self.rho_w
        self.d = 3000

        self.kx_grid, self.ky_grid = self._compute_wavenumbers()
        self._initialized = False

    def _compute_wavenumbers(self):
        """计算傅里叶域波数网格 (rad/m)，含纬度校正 cos(lat)。"""
        dx_deg = self.dx_deg
        dy_deg = self.region_size_deg / self.ny
        cos_lat = np.cos(np.radians(self.lat_center))
        dx_m = dx_deg * 111320 * cos_lat
        dy_m = dy_deg * 111320
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, dx_m)
        ky = 2 * np.pi * np.fft.fftfreq(self.ny, dy_m)
        return np.meshgrid(kx, ky)

    def gravity_forward(self, bathymetry, d=None, drho=None):
        """Parker 正演: 海底地形 → 重力异常 (mGal)。"""
        if d is None:
            d = self.d
        if drho is None:
            drho = self.drho

        k = np.sqrt(self.kx_grid ** 2 + self.ky_grid ** 2)
        k[0, 0] = 1e-10
        B_fft = np.fft.fft2(bathymetry)
        kernel = 2 * np.pi * self.G * drho * np.exp(-k * d)
        G_fft = B_fft * kernel
        gravity = np.real(np.fft.ifft2(G_fft))
        return gravity * 1e5

    def apply_flexural_isostasy(self, Te=10e3):
        """配置岩石圈挠曲均衡响应函数 Φ(k)，用于校正 Parker 反演核。

        Te=10km 是典型海洋岩石圈有效弹性厚度。
        """
        E = 7e10
        nu = 0.25
        g = 9.81
        rho_m = 3300

        D = E * Te ** 3 / (12 * (1 - nu ** 2))
        k = np.sqrt(self.kx_grid ** 2 + self.ky_grid ** 2)
        k[0, 0] = 1e-10

        phi = (D * k ** 4 + g * (rho_m - self.rho_c)) / (D * k ** 4 + g * (rho_m - self.rho_w))
        self.flexural_phi = phi
        print(f"Flexural isostasy configured (Te={Te/1000:.0f} km)")

    def apply_flexural_isostasy_from_age(self, age_grid, interp_order=1):
        """从洋壳年龄格网自动计算 Te 并配置挠曲均衡。

        使用冷却板模型: Te(km) = 3.0*sqrt(age) for age<40Ma, →35km for age>80Ma.
        取区域中位数 Te 作为标量代表值 (空间域变 Te 需解双调和方程, 暂不实现).

        Args:
            age_grid: 2D array, 洋壳年龄 [Ma], 0=陆地
            interp_order: 重采样到目标格网的插值阶数
        """
        from scipy.ndimage import zoom
        from utils import age_to_te
        sy = self.ny / age_grid.shape[0]
        sx = self.nx / age_grid.shape[1]
        age_resampled = zoom(np.nan_to_num(age_grid, nan=0),
                             (sy, sx), order=interp_order)
        te_km = age_to_te(age_resampled)
        te_median = np.nanmedian(te_km)
        if np.isnan(te_median) or te_median < 1:
            te_median = 10.0
            print(f"  Age grid has no valid ocean pixels — using default Te=10 km")
        te_m = te_median * 1000  # km → m
        self.apply_flexural_isostasy(Te=te_m)
        print(f"  Data-driven Te: median={te_median:.1f} km from crustal age")

    def apply_sediment_correction(self, sediment_thickness, drho_sed=500,
                                   interp_order=1):
        """沉积重力校正：从观测重力中减去沉积层的重力效应。

        沉积密度 (~2200 kg/m^3) 低于地壳密度 (~2670 kg/m^3)，
        厚沉积层区域的重力会被压低。减去沉积效应后，剩余重力
        仅反映基底地形，使 Parker 反演更准确。

        Args:
            sediment_thickness: 2D 沉积厚度 [m]
            drho_sed: 沉积-地壳密度差绝对值 [kg/m^3], |2200-2670| ≈ 470
            interp_order: 插值阶数 (GlobSed 5' → 目标格网)
        """
        from scipy.ndimage import zoom
        sy = self.ny / sediment_thickness.shape[0]
        sx = self.nx / sediment_thickness.shape[1]
        sed_resampled = zoom(np.nan_to_num(sediment_thickness, nan=0),
                             (sy, sx), order=interp_order)

        dg_sed = _sediment_gravity_correction(
            sed_resampled, drho_sed=drho_sed)

        self.gravity_obs = self.gravity_obs - dg_sed
        print(f"  Sediment correction: max_dg={np.nanmax(dg_sed):.1f} mGal, "
              f"mean_dg={np.nanmean(dg_sed[sed_resampled>0]):.1f} mGal "
              f"(drho={drho_sed} kg/m^3)")

    def gravity_to_bathymetry_linear(self, gravity, d=None, drho=None, k_min=None, k_max=None):
        """Parker 线性反演: 重力异常 → 海底地形，含 Tikhonov 正则化和挠曲校正。"""
        if d is None:
            d = self.d
        if drho is None:
            drho = self.drho

        k = np.sqrt(self.kx_grid ** 2 + self.ky_grid ** 2)
        k[0, 0] = 1e-10

        phi = getattr(self, 'flexural_phi', None)
        flex_corr = 1.0 / np.maximum(phi, 0.05) if phi is not None else 1.0

        reg = getattr(self, 'reg_param', 5.0)
        kernel_inv = np.exp(np.minimum(k * d, 50)) * flex_corr / (2 * np.pi * self.G * drho + reg * k ** 2)

        if k_min is not None:
            kernel_inv[k < k_min] = 0
        if k_max is not None:
            kernel_inv[k > k_max] = 0

        gravity_si = gravity * 1e-5
        G_fft = np.fft.fft2(gravity_si)
        B_fft = G_fft * kernel_inv
        bathymetry = np.real(np.fft.ifft2(B_fft))
        return bathymetry

    def calibrate_from_control_points(self, gravity_ship, bathy_ship):
        """从船载控制点标定 drho 和 d_ref。

        drho = median(|g|/|b|) / (2πG)·1e-5，d_ref = median(|b|)。
        """
        mask = ~(np.isnan(gravity_ship) | np.isnan(bathy_ship))
        if mask.sum() < 10:
            return
        g_s = gravity_ship[mask]
        b_s = bathy_ship[mask]

        grav_bathy_ratio = np.median(np.abs(g_s) / np.maximum(np.abs(b_s), 10))
        drho_est = grav_bathy_ratio * 1e-5 / (2 * np.pi * self.G)
        drho_est = max(500, min(3000, drho_est))
        self.drho = drho_est

        d_est = np.median(np.abs(b_s))
        d_est = max(500, min(6000, d_est))
        self.d = d_est

        print(f"  Calibrated drho={self.drho:.0f} kg/m^3, d_ref={self.d:.0f} m "
              f"from {mask.sum()} control points")

    def initialize(self, gravity_reconstructed, gravity_ship=None, bathy_ship=None,
                   bathymetry_prior=None, process_noise_std=50.0,
                   observation_noise_std=5.0, drho_prior=None, d_prior=None,
                   prior_mask=None):
        """初始化水深反演器：线性 Parker 反演 + GEBCO/船载先验融合。

        prior_mask: 布尔掩码，True=允许进入先验的区域 (如 TID=1 实测多波束区).
                   在掩码外区域的 GEBCO 值不会被混入，避免循环验证。
        """
        self.gravity_obs = gravity_reconstructed.copy()
        self.process_noise_std = process_noise_std
        self.obs_noise_std = observation_noise_std

        if drho_prior is not None:
            self.drho = drho_prior
        if d_prior is not None:
            self.d = d_prior
        else:
            if gravity_ship is not None and bathy_ship is not None:
                self.calibrate_from_control_points(gravity_ship, bathy_ship)
            else:
                grav_mean = np.abs(np.nanmean(gravity_reconstructed))
                d_est = grav_mean / (2 * np.pi * self.G * self.drho) * 1e-5
                self.d = max(1000, min(6000, d_est))
                print(f"  Auto-estimated reference depth: {self.d:.0f} m")

        self.reg_param = 5.0
        self.k_min = 2 * np.pi / 160000
        self.k_max = 2 * np.pi / 20000

        x0 = self.gravity_to_bathymetry_linear(
            gravity_reconstructed, d=self.d, drho=self.drho,
            k_min=self.k_min, k_max=self.k_max)

        if bathy_ship is not None:
            mask = ~np.isnan(bathy_ship)
            if mask.sum() > 50:
                bias = np.nanmedian(x0[mask] - bathy_ship[mask])
                x0 = x0 - bias
                x0[mask] = bathy_ship[mask]
                print(f"  Aligned to shipborne bathymetry, bias correction: {bias:.1f} m")

        if bathymetry_prior is not None:
            mask_prior = ~np.isnan(bathymetry_prior)
            if prior_mask is not None:
                mask_prior = mask_prior & prior_mask
            if bathy_ship is not None:
                mask_s = ~np.isnan(bathy_ship)
                blend_region = ~mask_s & mask_prior
                x0[blend_region] = 0.7 * x0[blend_region] + 0.3 * bathymetry_prior[blend_region]
                n_blend = blend_region.sum()
            else:
                blend_region = mask_prior
                x0[blend_region] = 0.5 * x0[blend_region] + 0.5 * bathymetry_prior[blend_region]
                n_blend = blend_region.sum()
            if n_blend > 0:
                tag = "TID=1" if prior_mask is not None else "all GEBCO"
                print(f"  GEBCO prior blended: {n_blend:,} pixels ({tag})")

        self.x = x0.ravel()
        self.P_diag = np.full(self.n_state, process_noise_std ** 2)
        self.k = np.sqrt(self.kx_grid ** 2 + self.ky_grid ** 2)
        self.k[0, 0] = 1e-10
        self._initialized = True

        print(f"Bathymetry inverter: d_ref={self.d:.0f} m, drho={self.drho:.0f} kg/m^3, "
              f"wavelength band=[{2*np.pi/self.k_max/1000:.0f}-{2*np.pi/self.k_min/1000:.0f}] km")

    def predict(self, smooth_sigma=1.5):
        """预测步：空间高斯平滑（水深连续性先验）。"""
        x_2d = self.x.reshape(self.ny, self.nx)
        x_pred = gaussian_filter(x_2d, sigma=smooth_sigma)
        self.x = x_pred.ravel()
        factor = 1.0 + smooth_sigma * 0.05
        self.P_diag = self.P_diag * factor + self.process_noise_std ** 2
        positive = self.P_diag[self.P_diag > 0]
        cap = 100.0 * (np.median(positive) + 1e-10) if positive.size > 0 else 1e10
        self.P_diag = np.minimum(self.P_diag, cap)

    def update_gravity(self, max_iter=50, lr=0.3):
        """迭代反演：用 Parker 正演残差成像更新水深估计。

        使用自适应步长 k_gain = lr*sqrt(P/(P+R))，等效于含协方差的梯度下降。
        """
        if not self._initialized:
            raise RuntimeError("请先调用 initialize() 初始化反演器")

        for it in range(max_iter):
            x_2d = self.x.reshape(self.ny, self.nx)
            gravity_pred = self.gravity_forward(x_2d, self.d, self.drho)
            residual = self.gravity_obs - gravity_pred
            jacobian_update = self.gravity_to_bathymetry_linear(
                residual, k_min=self.k_min, k_max=self.k_max)

            P_2d = self.P_diag.reshape(self.ny, self.nx)
            R = self.obs_noise_std ** 2
            k_gain_2d = lr * np.sqrt(np.maximum(P_2d, 0) / (P_2d + R + 1e-10))
            x_2d_updated = x_2d + k_gain_2d * jacobian_update

            d_min = getattr(self, 'depth_min', -11000)
            d_max = getattr(self, 'depth_max', 100)
            x_2d_updated = np.maximum(x_2d_updated, d_min)
            x_2d_updated = np.minimum(x_2d_updated, d_max)

            self.x = x_2d_updated.ravel()

            if it % 10 == 0:
                rmse = np.sqrt(np.mean(residual ** 2))
                if rmse < 0.1:
                    break

        return self.x.reshape(self.ny, self.nx)

    def run(self, n_iterations=5, smooth_sigma=1.0, inner_iter=20, verbose=True):
        """运行水深反演：预测-更新迭代。"""
        if not self._initialized:
            raise RuntimeError("请先调用 initialize() 初始化反演器")

        for it in range(n_iterations):
            self.predict(smooth_sigma=smooth_sigma)
            bathymetry = self.update_gravity(max_iter=inner_iter, lr=0.3)

            if verbose:
                print(f"  迭代 {it+1}/{n_iterations}: depth_mean={bathymetry.mean():.1f} m, "
                      f"range=[{bathymetry.min():.0f}, {bathymetry.max():.0f}]")

        return self.x.reshape(self.ny, self.nx)
