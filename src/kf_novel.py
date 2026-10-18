"""KF 地形物理先验 — 15'' 重力场重建主方法.

设计:
  15'' 网格的先验高频信息来自 GEBCO 地形布格正演 (物理上独立于 SSV 测高).
  SSV 1' 低频以 **块常数填充** (4×4 块内填同一值) 进入先验, 不做插值.
  块内 15'' 结构由 α × bathy_hp 提供, KF 空间平滑消除块边界阶跃.

数学形式:
  ssv_block = repeat_4x(ssv_1')                       # 块常数, 非插值
  prior = ssv_block + α × (bathy_bouguer - LP(bathy_bouguer))
  state init = prior, KF 用船载观测做后验修正

接口:
  build(gravity_lr, ship_15s, train_mask, lon_hr, lat_hr,
        lon_range, lat_range, upscale, bathy_15s) -> gravity_15s
"""
import numpy as np
from scipy.ndimage import gaussian_filter

from kalman_gravity import KalmanGravityReconstructor


# 物理常数
G_NEWTON = 6.6743e-11   # m^3 kg^-1 s^-2
RHO_CRUST = 2670.0      # kg/m^3  地壳平均密度
RHO_WATER = 1030.0      # kg/m^3  海水密度
DRHO_SUBMARINE = RHO_CRUST - RHO_WATER     # 1640 kg/m^3  海底地形界面的有效密度反差

# 一阶布格平板: Δg = 2πGΔρh, 单位 mGal/m
#   海底界面用 Δρ = ρ_c − ρ_w = 1640 (0.0688 mGal/m)
#   陆地界面用 Δρ = ρ_c − ρ_air ≈ 2670 (0.1118 mGal/m)
# 原实现统一用 ρ_c = 2670, 使海底起伏的重力效应被高估 2670/1640 = 1.63 倍,
# 也与本项目早期 Parker 反演代码里的 drho = 1640 自相矛盾。
BOUGUER_PER_M_OCEAN = 2 * np.pi * G_NEWTON * DRHO_SUBMARINE * 1e5   # 0.0688 mGal/m
BOUGUER_PER_M_LAND = 2 * np.pi * G_NEWTON * RHO_CRUST * 1e5         # 0.1118 mGal/m


def ocean_mask(bathy_15s):
    """True = 海底像元 (GEBCO 高程 < 0)。

    陆地像元的"海底地形—重力"关系不成立(上覆介质是空气而非海水),
    若把所有像元一视同仁, 海岸线处会出现虚假的强台阶。需要时用本掩膜剔除陆地。
    """
    return bathy_15s < 0


def block_fill(gravity_lr, upscale):
    """SSV 1' → 15'' 块常数填充 (非插值).

    每个 1' 像元值复制到 4×4 块内所有 15'' 像元, 不做任何插值.
    块内 15'' 结构由地形先验提供, KF 平滑消除块边界.
    """
    return np.repeat(np.repeat(gravity_lr, upscale, axis=0), upscale, axis=1)


class BathyPriorKF:
    """KF: 地形物理先验 — SSV 块常数 + GEBCO 布格重力高频.

    - SSV 1' 以块常数 (4×4 块内同值) 进入先验, 不做插值
    - 15'' 网格的先验高频信息来自地形正演 (布格近似)
    - 卫星测高无法分辨 15''-1' 频段, 但地形能; 这是物理上独立的信息源
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
        """GEBCO 高程 → 地形起伏引起的重力效应 (一阶布格平板近似, 单位 mGal)。

        海底界面 (bathy < 0): Δg = 2πG(ρ_c − ρ_w)·h   → 海沟为大负异常
        陆地界面 (bathy > 0): Δg = 2πG·ρ_c·h           → 山地为正异常

        注 1: 严格说这不是"布格重力异常", 而是地形/海底起伏的重力效应
              (相当于布格改正项); 与平均深度对应的常数项会被随后的高通去掉。
        注 2: 两种界面用不同的密度反差, 因此海岸线两侧是连续的物理量,
              不存在"陆地当海底"造成的虚假台阶。
        """
        coef = np.where(bathy_15s < 0, BOUGUER_PER_M_OCEAN, BOUGUER_PER_M_LAND)
        return coef * bathy_15s

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
              n_iter=6, smooth_sigma=4.0):
        """构建 KF 重力场.

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

        # SSV 低频以块常数填充 (4×4 块内同值, 完全无插值)
        ssv_block = block_fill(gravity_lr, upscale)

        # 地形布格重力高频 (核心物理先验)
        bouguer = self._bathy_to_bouguer(bathy_15s, lon_hr, lat_hr)
        bathy_hp = self._high_pass(bouguer, lon_hr, lat_hr)

        # 先验 = SSV 块常数 + α × 地形高频
        prior = ssv_block + self.alpha * bathy_hp

        print(f"    [KF] Bathy prior: "
              f"σ_ssv_block={ssv_block.std():.2f}, "
              f"σ_bathy_hp={bathy_hp.std():.2f} mGal, "
              f"α={self.alpha}, prior std={prior.std():.2f}")
        frac_ocean = float(np.mean(ocean_mask(bathy_15s)))
        print(f"         地形系数: 海底 {BOUGUER_PER_M_OCEAN:.4f} / 陆地 "
              f"{BOUGUER_PER_M_LAND:.4f} mGal/m; 海底像元占 {frac_ocean*100:.1f}%")
        print(f"         注: 海底界面改用 drho=rho_c-rho_w={DRHO_SUBMARINE:.0f} kg/m^3 后, "
              f"先验高频幅度变为原来的 {DRHO_SUBMARINE/RHO_CRUST:.3f} 倍, "
              f"α 需重新标定")

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
