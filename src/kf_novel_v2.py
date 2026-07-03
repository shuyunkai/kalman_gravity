"""三种改进 KF 方法 — 在 M2 基础上进一步突出与双线性的本质区别.

设计动机:
  M2 (BathyPriorKF) 用布格近似 g=2πGρh 做地形先验, 在大区域 α>1 时退化.
  原因: 布格近似忽略了 exp(-k·d) 向下延拓衰减, 短波长地形贡献被高估.

本文件实现三种改进:
  A. AdaptiveAlphaKF:   空间自适应 α, 船载稀疏区用大 α, 密集区用小 α
  B. ParkerPriorKF:     Parker 频域正演 (含 exp(-k·d) 衰减), 物理正确的地形先验
  C. MultiSourceKF:     频域三源 KF, SSV 低频 + Parker 中频 + 船载高频, 各频段独立约束

所有方法的核心仍是 KF: 状态在高分辨率网格上, 通过 predict-update 迭代融合观测.
"""
import numpy as np
from scipy.ndimage import zoom as sp_zoom, gaussian_filter
from scipy.fft import fft2, ifft2, fftfreq
from scipy.interpolate import RegularGridInterpolator

from kalman_gravity import KalmanGravityReconstructor

# 物理常数
G_NEWTON = 6.6743e-11  # m^3 kg^-1 s^-2
RHO_CRUST = 2670.0     # kg/m^3


# ============================================================
# 公共工具
# ============================================================

def high_pass_gaussian(field, dx_deg, dy_deg, cutoff_deg):
    """高斯高通滤波."""
    sigma_x = cutoff_deg / dx_deg
    sigma_y = cutoff_deg / dy_deg
    sigma = (sigma_x + sigma_y) / 2.0
    low = gaussian_filter(field, sigma=sigma)
    return field - low


def band_pass_gaussian(field, dx_deg, dy_deg, lo_deg, hi_deg):
    """高斯带通滤波 (lo_deg < wavelength < hi_deg)."""
    sigma_lo = lo_deg / ((dx_deg + dy_deg) / 2.0)
    sigma_hi = hi_deg / ((dx_deg + dy_deg) / 2.0)
    low = gaussian_filter(field, sigma=sigma_lo)
    high = gaussian_filter(field, sigma=sigma_hi)
    return low - high


def parker_forward_gravity(bathy_15s, dx_deg, dy_deg, lat_center,
                            rho=RHO_CRUST, d_mean=None):
    """Parker 频域正演: 从海底地形计算海面观测重力.

    公式: g(k) = 2πGρ × exp(-|k|·d) × F(h)
      h: 海底地形 (m, 正向下, 即深度)
      d: 平均海底深度 (m)
      exp(-|k|·d): 向下延拓衰减 (短波长信号在海面被严重衰减)

    与布格近似 (g=2πGρh) 区别:
      - 布格近似等价于 d=0, 即 exp(-|k|·d)=1
      - Parker 正演在短波长处指数衰减, 物理上正确
      - 这允许使用大 α 而不会让地形先验压过高频船载信号
    """
    # bathy_15s: GEBCO 海深 (负值), 转为正向深度 h
    h = -bathy_15s  # 正向下
    if d_mean is None:
        d_mean = float(np.mean(h[h > 0])) if np.any(h > 0) else 4000.0

    # 网格间距 (m), 考虑纬度
    dx_m = dx_deg * np.cos(np.radians(lat_center)) * 111000.0
    dy_m = dy_deg * 111000.0

    ny, nx = h.shape
    kx = 2 * np.pi * fftfreq(nx, d=dx_m)
    ky = 2 * np.pi * fftfreq(ny, d=dy_m)
    KX, KY = np.meshgrid(kx, ky)
    k = np.sqrt(KX ** 2 + KY ** 2)

    # Parker n=1 kernel (线性项, 含向下延拓衰减)
    kernel = 2 * np.pi * G_NEWTON * rho * np.exp(-k * d_mean)

    H = fft2(h)
    G = kernel * H
    g = np.real(ifft2(G))
    return g * 1e5  # m/s^2 → mGal


# ============================================================
# 方法 A: AdaptiveAlphaKF — 空间自适应 α
# ============================================================

class AdaptiveAlphaKF:
    """空间自适应 α KF: α(x,y) = f(船载点密度).

    数学形式:
      prior(x,y) = SSV(x,y) + α(x,y) × bathy_hp(x,y)
      α(x,y) = α_max × exp(-density(x,y) / scale)
        - 船载密集区: density 大 → α 小 (让 KF 主导)
        - 船载稀疏区: density 小 → α 大 (让地形先验补足)

    与双线性本质区别:
      - 先验是空间变化的, 不是均匀双线性
      - 稀疏区由物理地形主导 (而非双线性虚假高频)
      - 密集区由 KF 船载更新主导
    """

    def __init__(self, hr_grid_shape, lr_grid_shape, region_size_deg,
                 process_noise_std=2.0, sat_noise_std=3.0, ship_noise_std=1.0,
                 alpha_max=1.5, alpha_min=0.3, block_size_deg=1.0,
                 bathy_cutoff_deg=0.05):
        self.hr_shape = hr_grid_shape
        self.lr_shape = lr_grid_shape
        self.region_size_deg = region_size_deg
        self.Q = process_noise_std ** 2
        self.R_sat = sat_noise_std ** 2
        self.R_ship = ship_noise_std ** 2
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.block_size_deg = block_size_deg
        self.bathy_cutoff_deg = bathy_cutoff_deg

    def _compute_alpha_field(self, train_mask, lon_hr, lat_hr):
        """根据船载点密度计算空间变化 α 场."""
        # 块密度统计
        ny, nx = self.hr_shape
        dx = lon_hr[1] - lon_hr[0]
        dy = lat_hr[1] - lat_hr[0]
        block_size_pix_y = max(1, int(self.block_size_deg / dy))
        block_size_pix_x = max(1, int(self.block_size_deg / dx))
        nby = ny // block_size_pix_y
        nbx = nx // block_size_pix_x

        density = np.zeros((nby, nbx))
        for i in range(nby):
            for j in range(nbx):
                blk = train_mask[i*block_size_pix_y:(i+1)*block_size_pix_y,
                                  j*block_size_pix_x:(j+1)*block_size_pix_x]
                density[i, j] = blk.sum() / blk.size

        # α = α_max × exp(-density / scale), scale = median(density)
        non_zero = density[density > 0]
        scale = np.median(non_zero) if len(non_zero) > 0 else 0.01
        alpha_block = self.alpha_max * np.exp(-density / scale) + self.alpha_min
        # 上插到全网格
        alpha_field = np.repeat(np.repeat(alpha_block, block_size_pix_y, axis=0),
                                 block_size_pix_x, axis=1)
        # 补齐边界
        alpha_field = alpha_field[:ny, :nx]
        if alpha_field.shape != (ny, nx):
            tmp = np.full((ny, nx), self.alpha_min)
            tmp[:alpha_field.shape[0], :alpha_field.shape[1]] = alpha_field
            alpha_field = tmp
        # 平滑 α 场 (避免块边界突变)
        alpha_field = gaussian_filter(alpha_field, sigma=20)
        return alpha_field

    def build(self, gravity_lr, ship_15s, train_mask, lon_hr, lat_hr,
              lon_range, lat_range, upscale=4, bathy_15s=None,
              n_iter=6, smooth_sigma=15.0):
        if bathy_15s is None:
            raise ValueError("AdaptiveAlphaKF requires bathy_15s")

        # SSV 上采样
        ssv_hr = sp_zoom(gravity_lr, (upscale, upscale), order=1)

        # 布格重力 (作为地形高频源)
        bathy_bouguer = 2 * np.pi * G_NEWTON * RHO_CRUST * (-bathy_15s) * 1e5  # mGal
        dx_deg = lon_hr[1] - lon_hr[0]
        dy_deg = lat_hr[1] - lat_hr[0]
        bathy_hp = high_pass_gaussian(bathy_bouguer, dx_deg, dy_deg,
                                       self.bathy_cutoff_deg)

        # 空间变化 α 场
        alpha_field = self._compute_alpha_field(train_mask, lon_hr, lat_hr)
        print(f"    [A] α 场: range=[{alpha_field.min():.2f}, "
              f"{alpha_field.max():.2f}], mean={alpha_field.mean():.2f}")

        # 先验 = SSV + α(x,y) × bathy_hp
        prior = ssv_hr + alpha_field * bathy_hp

        # KF
        r = KalmanGravityReconstructor(
            grid_shape=self.hr_shape,
            region_size_deg=self.region_size_deg,
            lr_grid_shape=self.lr_shape)
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
# 方法 B: ParkerPriorKF — Parker 频域正演先验
# ============================================================

class ParkerPriorKF:
    """Parker 频域正演先验 KF.

    数学形式:
      g_parker = F⁻¹{2πGρ × exp(-|k|·d_mean) × F(h)} × 1e5  (mGal)
      prior = SSV + α × parker_hp
      KF 用船载观测修正

    与布格近似 (M2) 区别:
      - Parker 含 exp(-|k|·d) 衰减, 短波长信号在海面被指数压制
      - 物理正确: 海底 4km 深处 5km 波长的地形, 在海面只贡献 exp(-2π×4/5)≈0.006
      - 布格近似高估了短波长贡献, 所以 M2 在 α>1 时退化
      - Parker 支持大 α, 因为短波长本身就被衰减
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

    def build(self, gravity_lr, ship_15s, train_mask, lon_hr, lat_hr,
              lon_range, lat_range, upscale=4, bathy_15s=None,
              n_iter=6, smooth_sigma=15.0):
        if bathy_15s is None:
            raise ValueError("ParkerPriorKF requires bathy_15s")

        ssv_hr = sp_zoom(gravity_lr, (upscale, upscale), order=1)

        # Parker 正演 (核心物理改进)
        dx_deg = lon_hr[1] - lon_hr[0]
        dy_deg = lat_hr[1] - lat_hr[0]
        lat_center = (lat_range[0] + lat_range[1]) / 2.0
        g_parker = parker_forward_gravity(bathy_15s, dx_deg, dy_deg, lat_center)
        parker_hp = high_pass_gaussian(g_parker, dx_deg, dy_deg,
                                        self.bathy_cutoff_deg)

        print(f"    [B] Parker prior: σ_parker_hp={parker_hp.std():.2f} mGal "
              f"(vs 布格近似 σ~16 mGal), α={self.alpha}")

        prior = ssv_hr + self.alpha * parker_hp

        r = KalmanGravityReconstructor(
            grid_shape=self.hr_shape,
            region_size_deg=self.region_size_deg,
            lr_grid_shape=self.lr_shape)
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
# 方法 C: MultiSourceKF — 频域三源 KF
# ============================================================

class MultiSourceKF:
    """频域三源 KF: SSV 低频 + Parker 地形中频 + SSV 高频预填 + 船载修正.

    数学形式:
      prior = SSV_low(>50km) + α_mid × Parker_mid(8-50km) + SSV_high(<8km)
      KF 状态 = prior, 船载观测修正所有频段 (主要修正高频)

    设计思路:
      - 低频 (>50km): SSV 卫星测高最准, 直接用
      - 中频 (8-50km): 卫星分辨率不足, 用 Parker 地形正演替换 SSV 的中频
      - 高频 (<8km): 地形被衰减殆尽, 用 SSV 高频预填 (避免零初值导致 KF 收敛慢)
      - KF 用船载观测在所有频段上做后验修正

    与双线性本质区别:
      - 双线性: 单源 SSV, 中频信息是双线性插值的虚假高频
      - 三源 KF: 中频由物理正演提供 (与双线性虚假中频本质不同), 高频由 KF 修正
    """

    def __init__(self, hr_grid_shape, lr_grid_shape, region_size_deg,
                 process_noise_std=2.0, sat_noise_std=3.0, ship_noise_std=1.0,
                 alpha_mid=1.0, lo_cutoff_deg=0.5, hi_cutoff_deg=0.083):
        self.hr_shape = hr_grid_shape
        self.lr_shape = lr_grid_shape
        self.region_size_deg = region_size_deg
        self.Q = process_noise_std ** 2
        self.R_sat = sat_noise_std ** 2
        self.R_ship = ship_noise_std ** 2
        self.alpha_mid = alpha_mid
        self.lo_cutoff_deg = lo_cutoff_deg
        self.hi_cutoff_deg = hi_cutoff_deg

    def build(self, gravity_lr, ship_15s, train_mask, lon_hr, lat_hr,
              lon_range, lat_range, upscale=4, bathy_15s=None,
              n_iter=6, smooth_sigma=15.0):
        if bathy_15s is None:
            raise ValueError("MultiSourceKF requires bathy_15s")

        dx_deg = lon_hr[1] - lon_hr[0]
        dy_deg = lat_hr[1] - lat_hr[0]
        lat_center = (lat_range[0] + lat_range[1]) / 2.0
        sigma_factor = (dx_deg + dy_deg) / 2.0

        # SSV 全频段
        ssv_hr = sp_zoom(gravity_lr, (upscale, upscale), order=1)

        # 源 1: SSV 低频 (>50km)
        ssv_low = gaussian_filter(ssv_hr, sigma=self.lo_cutoff_deg / sigma_factor)
        print(f"    [C] Source 1 (SSV low, >{self.lo_cutoff_deg}°): "
              f"σ={ssv_low.std():.2f} mGal")

        # 源 2: Parker 地形 中频 (8-50km)
        g_parker = parker_forward_gravity(bathy_15s, dx_deg, dy_deg, lat_center)
        parker_mid = band_pass_gaussian(g_parker, dx_deg, dy_deg,
                                          self.hi_cutoff_deg, self.lo_cutoff_deg)
        print(f"    [C] Source 2 (Parker mid, {self.hi_cutoff_deg}-{self.lo_cutoff_deg}°): "
              f"σ={parker_mid.std():.2f} mGal")

        # 源 3: SSV 高频 (<8km) — 用 SSV 自己的高频预填, 避免零初值
        ssv_high = ssv_hr - gaussian_filter(ssv_hr, sigma=self.hi_cutoff_deg / sigma_factor)
        print(f"    [C] Source 3 (SSV high, <{self.hi_cutoff_deg}°): "
              f"σ={ssv_high.std():.2f} mGal (pre-fill, KF refines)")

        # 先验 = SSV 低频 + α × Parker 中频 + SSV 高频
        # 中频段: SSV 的中频被 Parker 的中频替换 (频率域三源融合)
        ssv_mid = ssv_hr - ssv_low - ssv_high
        prior = ssv_low + self.alpha_mid * parker_mid + ssv_high

        print(f"    [C] Prior: σ={prior.std():.2f} mGal "
              f"(SSV_mid σ={ssv_mid.std():.2f} replaced by Parker_mid × α)")

        r = KalmanGravityReconstructor(
            grid_shape=self.hr_shape,
            region_size_deg=self.region_size_deg,
            lr_grid_shape=self.lr_shape)
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
