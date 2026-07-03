"""M2 地形物理先验 KF — 15'' 重力场重建主方法.

设计动机:
  原始 KF (KalmanGravityReconstructor) 把 SSV 1' 双线性上采样作为状态初值,
  再用船载观测修正. 输出永远逃不开双线性的影子, 等价于"双线性 + 局部修正".

M2 (BathyPriorKF) 的本质改进:
  15" 网格的先验高频信息来自 GEBCO 地形布格正演 (物理上独立于 SSV 测高),
  而不是 SSV 1' 双线性插值. 卫星测高无法分辨 15"-1' 频段, 但地形能.

数学形式:
  prior = SSV_low_freq + α × (bathy_bouguer - low_pass(bathy_bouguer))
  state init = prior, KF 用船载观测做后验修正

接口:
  build(gravity_lr, ship_15s, train_mask, lon_hr, lat_hr,
        lon_range, lat_range, upscale, bathy_15s) -> gravity_15s
"""
import numpy as np
from scipy.ndimage import zoom as sp_zoom, gaussian_filter

from kalman_gravity import KalmanGravityReconstructor


# 物理常数
G_NEWTON = 6.6743e-11  # m^3 kg^-1 s^-2
RHO_CRUST = 2670.0     # kg/m^3 (地壳平均密度)

# 布格重力: g = 2πGρh, 对 h=1m: g = 0.1118 mGal
BOUGUER_PER_M = 2 * np.pi * G_NEWTON * RHO_CRUST * 1e5  # mGal per meter


class BathyPriorKF:
    """M2: 地形物理先验 KF — SSV 低频 + GEBCO 布格重力高频.

    与双线性本质区别:
      - 15" 网格的先验高频信息来自地形正演 (布格近似), 不是 SSV 插值
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
        """从 GEBCO 水深计算布格重力异常 (一阶布格近似).

        g_bouguer = 2π G ρ × depth (mGal)
        bathy 为海底深度 (负值, m), 海沟处为大负重力异常.
        """
        return BOUGUER_PER_M * bathy_15s

    def _high_pass(self, field, lon_hr, lat_hr):
        """高斯高通滤波, 保留 wavelength < cutoff_deg 的高频结构."""
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
        """构建 M2 重力场.

        参数:
          gravity_lr: SSV 1' 重力 (ny_lr, nx_lr)
          ship_15s: 船载重力格网化到 15'' (ny_hr, nx_hr), NaN 表示无观测
          train_mask: 训练点布尔掩码 (ny_hr, nx_hr)
          lon_hr, lat_hr: 15'' 经纬度向量
          lon_range, lat_range: 区域边界
          upscale: 上采样倍数 (1' → 15'' 为 4)
          bathy_15s: GEBCO 15'' 水深 (ny_hr, nx_hr), 必须提供
          n_iter: KF 迭代次数
          smooth_sigma: 空间平滑 sigma (15'' 像素)

        返回:
          gravity_15s: 15'' 重力场 (ny_hr, nx_hr)
        """
        if bathy_15s is None:
            raise ValueError("BathyPriorKF requires bathy_15s")

        # SSV 低频 (1' 双线性上采样, 仅作低频底色)
        ssv_hr = sp_zoom(gravity_lr, (upscale, upscale), order=1)

        # 地形布格重力高频 (核心物理先验)
        bouguer = self._bathy_to_bouguer(bathy_15s, lon_hr, lat_hr)
        bathy_hp = self._high_pass(bouguer, lon_hr, lat_hr)

        # 先验 = SSV + α × 地形高频
        prior = ssv_hr + self.alpha * bathy_hp

        print(f"    [M2] Bathy prior: σ_bathy_hp={bathy_hp.std():.2f} mGal, "
              f"α={self.alpha}, prior std={prior.std():.2f}")

        # KF 用此先验初始化 (绕过标准 initialize, 避免被船载点覆盖)
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
