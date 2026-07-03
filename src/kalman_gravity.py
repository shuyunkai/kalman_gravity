"""核心算法: 卡尔曼滤波重力场重建.

状态变量 x: 高分辨率格网 (ny_hr, nx_hr)
观测1 z_sat: 低分辨率卫星重力 (ny_lr, nx_lr), 观测模型为块均值
观测2 z_ship: 稀疏船载重力, 观测模型为点采样
状态转移: 空间高斯平滑, 模拟重力场连续性先验
"""
import numpy as np
from scipy.ndimage import gaussian_filter
import warnings

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
        P_2d = self.P_diag.reshape(self.ny, self.nx)
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
                P_block = P_2d[i0:i1, j0:j1].mean()
                K = P_block / (P_block + self.R_sat_val)
                innovation = z - z_pred
                block_flat_updated = block_flat + K * innovation
                x_2d[i0:i1, j0:j1] = block_flat_updated.reshape(block.shape)
                P_2d[i0:i1, j0:j1] = (1 - K) * P_block

        self.x = x_2d.ravel()
        self.P_diag = P_2d.ravel()  # explicit: P_2d is a view, ensure P_diag updated

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
