"""基线方法: 频域融合 (Butterworth 分带融合) 重建 15" 重力场.

参考论文:
  - ggad471 (Chen et al. 2024, GJI): Method 1 — Butterworth 分带融合
  - Marks & Smith 2016: 相干函数 = 0.5 定截止波长

核心思想:
    短波带 (< λc)  ← 船载 (高通)
    中长波 (≥ λc)  ← SSV   (低通)
    Δg_fused = HP(Δg_ship) + LP(Δg_ssv)

Butterworth 滤波器:
    H_LP(f) = 1 / (1 + (f/f_c)^{2n})
    H_HP(f) = (f/f_c)^{2n} / (1 + (f/f_c)^{2n})
    n = 4 (阶数), λc ≈ 27 km (相干 = 0.5 对应波长)

由于船载网格存在大量 NaN (覆盖率 ~17%), 需先用 SSV 填充再 FFT。
"""
import numpy as np
from scipy.ndimage import zoom as sp_zoom, gaussian_filter


def butterworth_2d_filter(data, dx_deg, dy_deg, lat_center, cutoff_km,
                          btype='low', order=4):
    """2D Butterworth 滤波器 (频域实现).

    Args:
        data: 2D array
        dx_deg, dy_deg: 经纬度方向网格间距 (度)
        lat_center: 中心纬度 (用于经度方向米/度换算)
        cutoff_km: 截止波长 (km)
        btype: 'low' or 'high'
        order: 滤波器阶数
    """
    ny, nx = data.shape
    cos_lat = np.cos(np.radians(lat_center))
    dx_m = dx_deg * 111320 * cos_lat
    dy_m = dy_deg * 111320

    kx = 2 * np.pi * np.fft.fftfreq(nx, dx_m)
    ky = 2 * np.pi * np.fft.fftfreq(ny, dy_m)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k = np.sqrt(kx_grid**2 + ky_grid**2)

    k_cutoff = 2 * np.pi / (cutoff_km * 1000)  # rad/m
    ratio = k / (k_cutoff + 1e-15)

    if btype == 'low':
        H = 1.0 / (1.0 + ratio**(2 * order))
    else:  # high
        H = ratio**(2 * order) / (1.0 + ratio**(2 * order))

    if btype == 'high':
        H[0, 0] = 0.0  # zero DC preserves mean

    data_fft = np.fft.fft2(data)
    filtered = np.real(np.fft.ifft2(data_fft * H))
    return filtered


def _estimate_coherence_cutoff(ship_filled, ssv_hr, dx_deg, dy_deg, lat_center):
    """估计相干函数 = 0.5 对应的截止波长 (ggad471 方法).

    Returns cutoff_km (默认回退 27 km).
    """
    try:
        ny, nx = ship_filled.shape
        cos_lat = np.cos(np.radians(lat_center))
        dx_m = dx_deg * 111320 * cos_lat
        dy_m = dy_deg * 111320

        kx = 2 * np.pi * np.fft.fftfreq(nx, dx_m)
        ky = 2 * np.pi * np.fft.fftfreq(ny, dy_m)
        kx_grid, ky_grid = np.meshgrid(kx, ky)
        k = np.sqrt(kx_grid**2 + ky_grid**2)

        F_s = np.fft.fft2(ship_filled)
        F_g = np.fft.fft2(ssv_hr)
        P_ss = np.abs(F_s)**2
        P_gg = np.abs(F_g)**2
        P_sg = F_s * np.conj(F_g)

        k_max = k.max()
        n_bins = 50
        k_bins = np.linspace(0, k_max, n_bins + 1)
        coh = np.zeros(n_bins)
        lam = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (k >= k_bins[i]) & (k < k_bins[i + 1])
            if mask.sum() < 10:
                continue
            ps = np.mean(P_ss[mask])
            pg = np.mean(P_gg[mask])
            psg = np.mean(P_sg[mask])
            coh[i] = np.abs(psg)**2 / (ps * pg + 1e-30)
            lam[i] = 2 * np.pi / np.sqrt(max(k_bins[i]**2, 1e-20))

        valid = lam > 0
        lam_v = lam[valid]
        coh_v = coh[valid]
        order = np.argsort(lam_v)
        lam_v = lam_v[order]
        coh_v = coh_v[order]
        idx = np.where(coh_v < 0.5)[0]
        if len(idx) > 0:
            cutoff_m = lam_v[idx[0]] * 1000  # km
            cutoff_m = float(np.clip(cutoff_m, 10, 60))
            print(f"  Estimated coherence=0.5 cutoff: {cutoff_m:.1f} km")
            return cutoff_m
    except Exception as e:
        print(f"  Coherence estimation failed ({e}), using default 27 km")
    return 27.0


def freq_fusion_reconstruct(ssv_hr, ship_grav, train_mask,
                             lon_hr, lat_hr, lon_range, lat_range,
                             cutoff_km=None, order=4, smooth_sigma=2.0):
    """频域融合 重建 15" 重力场 (干净接口).

    Args:
        ssv_hr: SSV 1' 双线性上采样到 15'' (ny_hr, nx_hr)
        ship_grav: 船载重力格网化到 15'' (ny_hr, nx_hr), NaN 表示无观测
        train_mask: 训练点布尔掩码 (ny_hr, nx_hr), 已做趋势恢复
        lon_hr, lat_hr: 15'' 经纬度向量
        lon_range, lat_range: 区域边界
        cutoff_km: 截止波长 (km). None 时自动估计相干=0.5 波长
        order: Butterworth 阶数
        smooth_sigma: 船载网格 Gaussian 平滑 sigma (像素), 抑制精确像素分配噪声

    Returns:
        gravity_fused: 15'' 融合重力场 (ny_hr, nx_hr)
    """
    # 船载网格 (仅训练点) → SSV 填充 NaN → 平滑
    ship_train_only = np.full_like(ship_grav, np.nan)
    ship_train_only[train_mask] = ship_grav[train_mask]
    nan_mask = np.isnan(ship_train_only)
    ship_filled = ship_train_only.copy()
    ship_filled[nan_mask] = ssv_hr[nan_mask]
    ship_filled = gaussian_filter(ship_filled, sigma=smooth_sigma)

    # SSV 平滑 (去除 1' 双线性上采样的子像素噪声)
    ssv_hr_smooth = gaussian_filter(ssv_hr, sigma=smooth_sigma)

    ny_hr, nx_hr = ssv_hr.shape
    dx_deg = (lon_range[1] - lon_range[0]) / nx_hr
    dy_deg = (lat_range[1] - lat_range[0]) / ny_hr
    lat_center = (lat_range[0] + lat_range[1]) / 2.0

    if cutoff_km is None:
        cutoff_km = _estimate_coherence_cutoff(
            ship_filled, ssv_hr_smooth, dx_deg, dy_deg, lat_center)
    print(f"  Freq fusion: cutoff={cutoff_km:.1f} km, order={order}, "
          f"smooth_sigma={smooth_sigma}")

    ship_hp = butterworth_2d_filter(ship_filled, dx_deg, dy_deg, lat_center,
                                     cutoff_km, btype='high', order=order)
    ssv_lp = butterworth_2d_filter(ssv_hr_smooth, dx_deg, dy_deg, lat_center,
                                    cutoff_km, btype='low', order=order)
    return ssv_lp + ship_hp
