"""BLA-KF: Block-observation Localized Adaptive Kalman Filter.

块观测（1′ 足迹）+ 协方差局部化（集合相关长度 ℓ≈28 px）+ 自适应 R_sat。
用于 15″ 重力场重建；R_ship 固定 10 mGal，地形先验 α=0.5。
"""
import numpy as np

from kf_novel import BathyPriorKF


class BLAKF(BathyPriorKF):
    """15″ 生产配置：P0=Q=16, ℓ=28 px, adaptive R_sat, α=0.5."""

    def __init__(self, hr_grid_shape, lr_grid_shape, region_size_deg,
                 alpha=0.5, ship_noise_std=10.0, sat_noise_std=3.0,
                 corr_px=28.0):
        super().__init__(
            hr_grid_shape, lr_grid_shape, region_size_deg,
            process_noise_std=4.0, sat_noise_std=sat_noise_std,
            ship_noise_std=ship_noise_std, alpha=alpha,
            p0_std=4.0, q_std=4.0, adaptive_ship=False)
        self.corr_px = corr_px

    def build(self, gravity_lr, ship_15s, train_mask, lon_hr, lat_hr,
              lon_range, lat_range, upscale=4, bathy_15s=None,
              n_iter=6, smooth_sigma=4.0):
        return super().build(
            gravity_lr, ship_15s, train_mask, lon_hr, lat_hr,
            lon_range, lat_range, upscale=upscale, bathy_15s=bathy_15s,
            n_iter=n_iter, smooth_sigma=smooth_sigma,
            localize=True, adaptive=True, corr_px=self.corr_px)
