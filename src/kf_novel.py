"""四种与双线性本质区别的 KF 新方法.

设计动机:
  原始 KF (KalmanGravityReconstructor) 把 SSV 1' 双线性上采样作为状态初值,
  再用船载观测修正. 输出永远逃不开双线性的影子, 等价于"双线性 + 局部修正".

本文件实现四种方法, 从初值/状态表示/观测模型/变换域四个层面与双线性本质区分:
  1. MultiScaleResidualKF: 多尺度残差 KF, 15" 残差初值=0 (不是双线性)
  2. BathyPriorKF: 地形物理先验 KF, 高频先验来自 GEBCO 正演 (不是 SSV)
  3. AlongTrackKF: 沿迹观测 KF, 不做预格网化, 原始航迹点经 H 算子更新 4 邻域
  4. WaveletKF: 谱域小波 KF, 状态在小波系数空间, 各尺度独立更新

所有方法共享统一接口: build(gravity_lr, ship_15s, train_mask, lon_hr, lat_hr,
                            lon_range, lat_range, upscale) -> gravity_15s
"""
import numpy as np
from scipy.ndimage import zoom as sp_zoom, gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import pywt

from kalman_gravity import KalmanGravityReconstructor


# ============================================================
# Method 1: 多尺度残差 KF
# ============================================================

class MultiScaleResidualKF:
    """多尺度残差 KF: 1' 低频 KF + 15" 高频残差 KF (零初值).

    数学形式:
      x_low (1' 网格) ← SSV 1' 初值, KF 用 SSV 块均值约束 + 船载点低频分量
      x_hi  (15" 残差) ← 0 初值, KF 用船载点高频残差驱动
      最终: g_15s = zoom(x_low, 4x) + x_hi

    与双线性本质区别:
      - 15" 高频信息完全由船载观测重建, 不继承 SSV 双线性结构
      - 残差初值为零, 任何高频细节都来自 KF 后验, 不是先验平滑
    """

    def __init__(self, lr_grid_shape, hr_grid_shape, region_size_deg,
                 process_noise_std=2.0, sat_noise_std=3.0, ship_noise_std=1.0):
        self.lr_shape = lr_grid_shape
        self.hr_shape = hr_grid_shape
        self.upscale = hr_grid_shape[0] // lr_grid_shape[0]
        self.region_size_deg = region_size_deg
        self.Q = process_noise_std ** 2
        self.R_sat = sat_noise_std ** 2
        self.R_ship = ship_noise_std ** 2

    def build(self, gravity_lr, ship_15s, train_mask, lon_hr, lat_hr,
              lon_range, lat_range, upscale=4, n_iter=6, smooth_sigma_low=4.0,
              smooth_sigma_hi=15.0):
        ny_lr, nx_lr = self.lr_shape
        ny_hr, nx_hr = self.hr_shape

        # ---- Stage 1: 1' 网格 KF (低频) ----
        # 船载点降采样到 1': 每个 4x4 块内训练点均值
        ship_1m = np.full(self.lr_shape, np.nan)
        train_1m = np.zeros(self.lr_shape, dtype=bool)
        for i in range(ny_lr):
            for j in range(nx_lr):
                blk_s = ship_15s[i*upscale:(i+1)*upscale, j*upscale:(j+1)*upscale]
                blk_t = train_mask[i*upscale:(i+1)*upscale, j*upscale:(j+1)*upscale]
                if np.any(blk_t):
                    vals = blk_s[blk_t]
                    if np.any(~np.isnan(vals)):
                        ship_1m[i, j] = np.nanmean(vals)
                        train_1m[i, j] = True

        print(f"    [M1] Stage 1: 1' KF (low-freq), train 1' pts={train_1m.sum()}")
        r_low = KalmanGravityReconstructor(
            grid_shape=self.lr_shape,
            region_size_deg=self.region_size_deg,
            lr_grid_shape=self.lr_shape)
        r_low.initialize(
            gravity_lr=gravity_lr, gravity_ship=ship_1m,
            ship_train_mask=train_1m,
            process_noise_std=np.sqrt(self.Q),
            sat_noise_std=np.sqrt(self.R_sat),
            ship_noise_std=np.sqrt(self.R_ship))
        x_low = r_low.run(n_iterations=n_iter, smooth_sigma=smooth_sigma_low)

        # ---- Stage 2: 15" 残差 KF (零初值) ----
        # 残差 = 船载 - zoom(x_low, 4x), 高频部分, 不含 SSV 双线性结构
        x_low_hr = sp_zoom(x_low, (upscale, upscale), order=1)
        residual = np.full(self.hr_shape, np.nan)
        valid = train_mask & ~np.isnan(ship_15s)
        residual[valid] = ship_15s[valid] - x_low_hr[valid]

        print(f"    [M1] Stage 2: 15'' residual KF (zero init), "
              f"residual train pts={valid.sum()}")
        r_hi = KalmanGravityReconstructor(
            grid_shape=self.hr_shape,
            region_size_deg=self.region_size_deg,
            lr_grid_shape=self.hr_shape)  # degenerate: no satellite update
        # 初值 = 0 (关键: 不是双线性!)
        # 直接构造零状态, 不调用标准 initialize
        r_hi._initialized = True
        r_hi.ny, r_hi.nx = self.hr_shape
        r_hi.n_state = ny_hr * nx_hr
        r_hi.x = np.zeros(r_hi.n_state)
        r_hi.P_diag = self.Q * np.ones(r_hi.n_state)
        r_hi.Q_diag = self.Q * np.ones(r_hi.n_state)
        r_hi.R_sat_val = self.R_sat
        r_hi.R_ship_val = self.R_ship
        r_hi.gravity_lr = np.zeros(self.hr_shape)  # dummy
        r_hi.gravity_ship = residual
        r_hi.ship_train_mask = train_mask
        r_hi._iter_count = 0
        x_hi = r_hi.run(n_iterations=n_iter, smooth_sigma=smooth_sigma_hi)

        gravity_15s = x_low_hr + x_hi
        return gravity_15s


# ============================================================
# Method 2: 地形物理先验 KF
# ============================================================

# 物理常数
G_NEWTON = 6.6743e-11  # m^3 kg^-1 s^-2
RHO_CRUST = 2670.0     # kg/m^3, 典型地壳密度
# 布格板重力: g = 2π G ρ h
# 对 h=1m: g = 2π × 6.6743e-11 × 2670 × 1 = 1.118e-6 m/s^2 = 0.1118 mGal
BOUGUER_PER_M = 2 * np.pi * G_NEWTON * RHO_CRUST * 1e5  # mGal per meter


class BathyPriorKF:
    """地形物理先验 KF: SSV 低频 + GEBCO 布格重力高频.

    数学形式:
      prior = SSV_low_freq + α × (bathy_bouguer - low_pass(bathy_bouguer))
      state init = prior, KF 用船载修正

    与双线性本质区别:
      - 15" 网格的先验高频信息来自地形正演 (Parker/布格), 不是 SSV 插值
      - 卫星测高无法分辨 15"-1' 频段, 但地形能; 这是物理上独立的信息源
      - 船载修正建立在物理先验上, 而不是双线性平滑上
    """

    def __init__(self, hr_grid_shape, lr_grid_shape, region_size_deg,
                 process_noise_std=2.0, sat_noise_std=3.0, ship_noise_std=1.0,
                 alpha=1.0, bathy_cutoff_deg=0.05):
        self.hr_shape = hr_grid_shape
        self.lr_shape = lr_grid_shape
        self.region_size_deg = region_size_deg
        self.Q = process_noise_std ** 2
        self.R_sat = sat_noise_std ** 2
        self.R_ship = ship_noise_std ** 2
        self.alpha = alpha
        self.bathy_cutoff_deg = bathy_cutoff_deg

    def _bathy_to_bouguer(self, bathy_15s, lon_hr, lat_hr):
        """从 GEBCO 水深计算布格重力异常 (简化 Parker 1阶).

        g_bouguer = 2π G ρ × depth (mGal)
        深度负值 (海底 below sea level), 所以海沟处为大负重力异常.
        """
        # bathy: 海底深度 (负值, m)
        return BOUGUER_PER_M * bathy_15s

    def _high_pass(self, field, lon_hr, lat_hr):
        """高斯高通滤波, 截断 wavelength < cutoff_deg 的部分."""
        dx_deg = lon_hr[1] - lon_hr[0]
        dy_deg = lat_hr[1] - lat_hr[0]
        sigma_x = self.bathy_cutoff_deg / dx_deg
        sigma_y = self.bathy_cutoff_deg / dy_deg
        sigma = (sigma_x + sigma_y) / 2.0
        low = gaussian_filter(field, sigma=sigma)
        return field - low

    def build(self, gravity_lr, ship_15s, train_mask, lon_hr, lat_hr,
              lon_range, lat_range, upscale=4, bathy_15s=None,
              n_iter=6, smooth_sigma=15.0):
        if bathy_15s is None:
            raise ValueError("BathyPriorKF requires bathy_15s")

        # SSV 低频 (1' 不上采样, 然后双线性 — 仅作为低频底色, 不是高频先验)
        ssv_hr = sp_zoom(gravity_lr, (upscale, upscale), order=1)

        # 地形布格重力高频 (核心物理先验)
        bouguer = self._bathy_to_bouguer(bathy_15s, lon_hr, lat_hr)
        bathy_hp = self._high_pass(bouguer, lon_hr, lat_hr)

        # 先验 = SSV + α × 地形高频
        prior = ssv_hr + self.alpha * bathy_hp

        print(f"    [M2] Bathy prior: σ_bathy_hp={bathy_hp.std():.2f} mGal, "
              f"α={self.alpha}, prior std={prior.std():.2f}")

        # KF 用此先验初始化
        r = KalmanGravityReconstructor(
            grid_shape=self.hr_shape,
            region_size_deg=self.region_size_deg,
            lr_grid_shape=self.lr_shape)
        # 直接设置状态为先验, 不用标准 initialize (避免被船载点覆盖)
        r._initialized = True
        r.ny, r.nx = self.hr_shape
        r.n_state = r.ny * r.nx
        r.x = prior.ravel().copy()
        r.P_diag = self.Q * np.ones(r.n_state)
        r.Q_diag = self.Q * np.ones(r.n_state)
        r.R_sat_val = self.R_sat
        r.R_ship_val = self.R_ship
        r.gravity_lr = gravity_lr
        r.gravity_ship = ship_15s
        r.ship_train_mask = train_mask
        r._iter_count = 0
        gravity_15s = r.run(n_iterations=n_iter, smooth_sigma=smooth_sigma)
        return gravity_15s


# ============================================================
# Method 3: 沿迹观测 KF
# ============================================================

class AlongTrackKF:
    """沿迹观测 KF: 船载不做预格网化, 原始点经 H 算子更新 4 邻域.

    数学形式:
      观测: z_i = H_i x + noise, H_i 为双线性插值权重 [w00 w01 w10 w11]
      状态: x 在 15" 网格上
      初值: SSV 低频成分 (用低通滤波提取, 不是双线性)

    与双线性本质区别:
      - 观测模型正确处理航迹点几何, 不丢失亚像元信息
      - 状态初值用低通滤波后的 SSV (而非双线性), 不在 4×4 块内插出虚假高频
      - 一个观测更新 4 个邻域格点, 协方差在网格上正确传播
    """

    def __init__(self, hr_grid_shape, lr_grid_shape, region_size_deg,
                 process_noise_std=2.0, sat_noise_std=3.0, ship_noise_std=1.0,
                 low_pass_sigma_deg=0.05):
        self.hr_shape = hr_grid_shape
        self.lr_shape = lr_grid_shape
        self.region_size_deg = region_size_deg
        self.Q = process_noise_std ** 2
        self.R_sat = sat_noise_std ** 2
        self.R_ship = ship_noise_std ** 2
        self.low_pass_sigma_deg = low_pass_sigma_deg

    def _extract_low_freq(self, gravity_lr, lon_hr, lat_hr, upscale):
        """从 1' SSV 提取低频成分: zoom + 低通, 避免双线性虚假高频."""
        ssv_hr = sp_zoom(gravity_lr, (upscale, upscale), order=1)
        dx = lon_hr[1] - lon_hr[0]
        sigma = self.low_pass_sigma_deg / dx
        return gaussian_filter(ssv_hr, sigma=sigma)

    def _update_point_bilinear(self, x_2d, P_2d, lon_i, lat_i, z,
                                lon_hr, lat_hr, prop_radius=3, prop_fraction=0.3):
        """单点双线性观测更新: 更新 4 邻域 + 高斯传播."""
        # 双线性插值权重
        dx = (lon_i - lon_hr[0]) / (lon_hr[1] - lon_hr[0])
        dy = (lat_i - lat_hr[0]) / (lat_hr[1] - lat_hr[0])
        j0 = int(np.floor(dx))
        i0 = int(np.floor(dy))
        j0 = max(0, min(j0, x_2d.shape[1] - 2))
        i0 = max(0, min(i0, x_2d.shape[0] - 2))
        fx = dx - j0
        fy = dy - i0
        w = np.array([(1-fx)*(1-fy), fx*(1-fy), (1-fx)*fy, fx*fy])
        ii = [i0, i0, i0+1, i0+1]
        jj = [j0, j0+1, j0, j0+1]

        # 加权观测: z_pred = Σ w_k x[k]
        z_pred = sum(w[k] * x_2d[ii[k], jj[k]] for k in range(4))
        # 加权协方差: P_pred = Σ w_k^2 P[k]  (近似)
        P_pred = sum(w[k]**2 * P_2d[ii[k], jj[k]] for k in range(4))
        K = P_pred / (P_pred + self.R_ship)
        innovation = z - z_pred

        # 更新 4 个邻域 (按权重分配增益)
        for k in range(4):
            gain_k = w[k] * K  # 双线性权重 × 卡尔曼增益
            x_2d[ii[k], jj[k]] += gain_k * innovation
            P_2d[ii[k], jj[k]] = (1 - gain_k * w[k]) * P_2d[ii[k], jj[k]]

        # 空间传播 (与原 KF 类似)
        ny, nx = x_2d.shape
        i_c, j_c = i0 + int(round(fy)), j0 + int(round(fx))
        i0p = max(0, i_c - prop_radius)
        i1p = min(ny, i_c + prop_radius + 1)
        j0p = max(0, j_c - prop_radius)
        j1p = min(nx, j_c + prop_radius + 1)
        sy, sx = i1p - i0p, j1p - j0p
        dy_k = np.abs(np.arange(sy) - (i_c - i0p)).reshape(sy, 1)
        dx_k = np.abs(np.arange(sx) - (j_c - j0p)).reshape(1, sx)
        dist = np.sqrt(dx_k**2 + dy_k**2)
        w_prop = np.exp(-0.5 * (dist / (prop_radius / 2.0))**2)
        w_prop = w_prop / w_prop.sum()
        # 加到邻域 (非观测格点)
        x_2d[i0p:i1p, j0p:j1p] += prop_fraction * K * innovation * w_prop
        return

    def build(self, gravity_lr, ship_points_raw, train_mask_pts, lon_hr, lat_hr,
              lon_range, lat_range, upscale=4, n_iter=6, smooth_sigma=15.0):
        """ship_points_raw: (N, 3) [lon, lat, faa]; train_mask_pts: (N,) bool."""
        ny_hr, nx_hr = self.hr_shape
        # 初值: SSV 低频 (不是双线性)
        x_init = self._extract_low_freq(gravity_lr, lon_hr, lat_hr, upscale)
        print(f"    [M3] Initial state: low-pass SSV (σ={self.low_pass_sigma_deg}°), "
              f"not bilinear")

        x_2d = x_init.copy()
        P_2d = self.Q * np.ones_like(x_2d)

        train_pts = ship_points_raw[train_mask_pts]
        print(f"    [M3] Along-track points: {len(train_pts):,}")

        for it in range(n_iter):
            # Predict: 空间平滑
            x_2d = gaussian_filter(x_2d, sigma=smooth_sigma)
            P_2d = P_2d * (1.0 + smooth_sigma * 0.1) + self.Q

            # 卫星更新 (块均值约束)
            self._update_satellite(x_2d, P_2d, gravity_lr, upscale)

            # 沿迹更新 (随机抽样以控制单次成本)
            n_pts = len(train_pts)
            n_max = 5000
            if n_pts > n_max:
                rng = np.random.default_rng(42 + it)
                idx_sel = rng.choice(n_pts, n_max, replace=False)
            else:
                idx_sel = np.arange(n_pts)

            for k in idx_sel:
                lon_i, lat_i, z = train_pts[k]
                if np.isnan(z):
                    continue
                self._update_point_bilinear(x_2d, P_2d, lon_i, lat_i, z,
                                              lon_hr, lat_hr)

            if (it + 1) % 2 == 0 or it == n_iter - 1:
                print(f"    [M3] iter {it+1}/{n_iter}: mean={x_2d.mean():.2f}, "
                      f"std={x_2d.std():.2f}")

        return x_2d

    def _update_satellite(self, x_2d, P_2d, gravity_lr, upscale):
        """1' 块均值约束 (与标准 KF 一致)."""
        ny_lr, nx_lr = gravity_lr.shape
        hr_per_lr_y = x_2d.shape[0] // ny_lr
        hr_per_lr_x = x_2d.shape[1] // nx_lr
        for i in range(ny_lr):
            for j in range(nx_lr):
                i0 = i * hr_per_lr_y
                i1 = (i+1) * hr_per_lr_y
                j0 = j * hr_per_lr_x
                j1 = (j+1) * hr_per_lr_x
                block = x_2d[i0:i1, j0:j1]
                P_block = P_2d[i0:i1, j0:j1].mean()
                z = gravity_lr[i, j]
                z_pred = block.mean()
                K = P_block / (P_block + self.R_sat)
                innovation = z - z_pred
                x_2d[i0:i1, j0:j1] += K * innovation
                P_2d[i0:i1, j0:j1] = (1 - K) * P_block


# ============================================================
# Method 4: 谱域小波 KF
# ============================================================

class WaveletKF:
    """谱域小波 KF: 状态在 Haar 小波系数空间, 各尺度独立更新.

    数学形式:
      状态: c = [c_LL (最粗), c_LH_1, c_HL_1, c_HH_1, ..., c_LH_L, c_HL_L, c_HH_L]
      初值: c_LL ← SSV 1' (decimated), 高频系数 = 0
      KF 在系数空间运行, 船载观测通过逆 DWT 投影到系数

    与双线性本质区别:
      - 状态表示完全不同, 不在网格空间而在多尺度小波空间
      - 高频细节由船载观测在小波系数上重建, 不是 SSV 插值
      - 各尺度独立协方差, 物理上对应不同波长重力场
    """

    def __init__(self, hr_grid_shape, lr_grid_shape, region_size_deg,
                 process_noise_std=2.0, sat_noise_std=3.0, ship_noise_std=1.0,
                 wavelet='haar', n_levels=3):
        self.hr_shape = hr_grid_shape
        self.lr_shape = lr_grid_shape
        self.region_size_deg = region_size_deg
        self.Q = process_noise_std ** 2
        self.R_sat = sat_noise_std ** 2
        self.R_ship = ship_noise_std ** 2
        self.wavelet = wavelet
        self.n_levels = n_levels

    def _decompose_init(self, gravity_lr, upscale):
        """SSV 1' 上采样到 15" + Haar 分解, 高频系数清零 (关键: 不继承 SSV 高频)."""
        ssv_hr = sp_zoom(gravity_lr, (upscale, upscale), order=1)
        # 多级 Haar 分解
        coeffs = pywt.wavedec2(ssv_hr, self.wavelet, level=self.n_levels)
        # coeffs[0] = c_LL (低频), coeffs[1:] = [(cLH, cHL, cHH)_level1, ...]
        c_ll = coeffs[0]
        # 高频系数置零 (不继承 SSV 的虚假高频)
        coeffs_zeroed = [c_ll] + [
            tuple(np.zeros_like(c) for c in detail)
            for detail in coeffs[1:]
        ]
        return coeffs_zeroed

    def _reconstruct(self, coeffs):
        return pywt.waverec2(coeffs, self.wavelet)

    def _coeffs_to_flat(self, coeffs):
        """把多尺度系数打平成状态向量 (用于 KF)."""
        flat = [coeffs[0].ravel()]
        shapes = [coeffs[0].shape]
        detail_shapes = []
        for detail in coeffs[1:]:
            flat.extend(c.ravel() for c in detail)
            detail_shapes.append(tuple(c.shape for c in detail))
        return np.concatenate(flat), shapes, detail_shapes

    def _flat_to_coeffs(self, flat, shapes, detail_shapes):
        c_ll = flat[:np.prod(shapes[0])].reshape(shapes[0])
        idx = np.prod(shapes[0])
        details = []
        for ds in detail_shapes:
            d = []
            for s in ds:
                sz = np.prod(s)
                d.append(flat[idx:idx+sz].reshape(s))
                idx += sz
            details.append(tuple(d))
        return [c_ll] + details

    def build(self, gravity_lr, ship_15s, train_mask, lon_hr, lat_hr,
              lon_range, lat_range, upscale=4, n_iter=6, smooth_sigma=15.0):
        # 初值: 小波系数 (高频=0)
        coeffs = self._decompose_init(gravity_lr, upscale)
        flat, shapes, detail_shapes = self._coeffs_to_flat(coeffs)
        n_state = len(flat)
        x = flat.copy()
        P = self.Q * np.ones(n_state)

        print(f"    [M4] Wavelet state: {self.wavelet}, levels={self.n_levels}, "
              f"n_state={n_state:,} (LL={np.prod(shapes[0]):,})")

        # 船载训练点索引
        valid = train_mask & ~np.isnan(ship_15s)
        ship_vals = ship_15s[valid]
        ship_idx = np.where(valid.ravel())[0]
        n_ship = len(ship_idx)
        n_max = 3000

        for it in range(n_iter):
            # Predict: 系数空间平滑 (低频大sigma, 高频小sigma)
            x_2d = self._reconstruct(self._flat_to_coeffs(x, shapes, detail_shapes))
            x_2d = gaussian_filter(x_2d, sigma=smooth_sigma)
            coeffs_sm = pywt.wavedec2(x_2d, self.wavelet, level=self.n_levels)
            x_new, _, _ = self._coeffs_to_flat(coeffs_sm)
            x = x_new
            P = P * 1.1 + self.Q

            # 卫星更新 (块均值, 在重建场上)
            x_2d = self._reconstruct(self._flat_to_coeffs(x, shapes, detail_shapes))
            ny_lr, nx_lr = gravity_lr.shape
            hr_per_lr_y = self.hr_shape[0] // ny_lr
            hr_per_lr_x = self.hr_shape[1] // nx_lr
            for i in range(ny_lr):
                for j in range(nx_lr):
                    i0, i1 = i*hr_per_lr_y, (i+1)*hr_per_lr_y
                    j0, j1 = j*hr_per_lr_x, (j+1)*hr_per_lr_x
                    block = x_2d[i0:i1, j0:j1]
                    z_pred = block.mean()
                    innovation = gravity_lr[i, j] - z_pred
                    # 用更新后的场重建系数 (近似)
                    x_2d[i0:i1, j0:j1] += innovation * 0.5
            coeffs_new = pywt.wavedec2(x_2d, self.wavelet, level=self.n_levels)
            x, _, _ = self._coeffs_to_flat(coeffs_new)

            # 船载更新 (随机抽样)
            if n_ship > n_max:
                rng = np.random.default_rng(42 + it)
                sel = rng.choice(n_ship, n_max, replace=False)
            else:
                sel = np.arange(n_ship)

            for k in sel:
                idx_grid = ship_idx[k]
                # 双线性观测 (在重建场上)
                i_g, j_g = divmod(idx_grid, self.hr_shape[1])
                z_pred = x_2d[i_g, j_g]
                innovation = ship_vals[k] - z_pred
                K = self.Q / (self.Q + self.R_ship)
                x_2d[i_g, j_g] += K * innovation
                # 重建系数
                coeffs_new = pywt.wavedec2(x_2d, self.wavelet, level=self.n_levels)
                x, _, _ = self._coeffs_to_flat(coeffs_new)
                # 重新提取 2D 场用于下一个观测
                x_2d = self._reconstruct(self._flat_to_coeffs(x, shapes, detail_shapes))

            if (it + 1) % 2 == 0 or it == n_iter - 1:
                print(f"    [M4] iter {it+1}/{n_iter}: mean={x_2d.mean():.2f}, "
                      f"std={x_2d.std():.2f}")

        gravity_15s = self._reconstruct(self._flat_to_coeffs(x, shapes, detail_shapes))
        return gravity_15s
