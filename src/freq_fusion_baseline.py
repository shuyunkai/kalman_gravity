"""基线方法 B: 频域融合 重建 15" 重力场。

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
import sys, time
from pathlib import Path
import numpy as np
from scipy.ndimage import zoom as sp_zoom, gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))

from run_pipeline import load_ssv_region, load_shipborne_gravity, NC_ENGINE, DATA_DIR
from utils import split_shipborne_mask, compute_metrics

OUTPUT_DIR = Path(__file__).parent.parent / "output_real"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _residual_to_faa(ship_grav, lon_range, lat_range, lon_lr, lat_lr, gravity_lr, upscale):
    """复用 run_pipeline 的 EGM2008 趋势恢复逻辑 (与 kriging_baseline 相同)。"""
    ship_mask = ~np.isnan(ship_grav)
    if ship_mask.sum() == 0:
        return ship_grav
    before_mean = np.nanmean(ship_grav[ship_mask])
    if abs(before_mean) >= 50:
        print(f"  Shipborne already FAA (mean={before_mean:.0f} mGal), skip trend recovery")
        return ship_grav

    try:
        ds_egm = xr.open_dataset(
            DATA_DIR / "icgem" / "grid" / "EGM2008_gravity_anomaly.nc",
            engine=NC_ENGINE)
        vname = [v for v in ds_egm.variables if len(ds_egm[v].dims) >= 2][0]
        grav_egm = ds_egm[vname].sel(
            lon=slice(lon_range[0], lon_range[1]),
            lat=slice(lat_range[1], lat_range[0])).values
        ds_egm.close()
        egm_trend = gaussian_filter(grav_egm, sigma=10)
        xi_egm = np.linspace(lon_range[0], lon_range[1], egm_trend.shape[1])
        yi_egm = np.linspace(lat_range[1], lat_range[0], egm_trend.shape[0])
        interp = RegularGridInterpolator(
            (yi_egm, xi_egm), egm_trend, bounds_error=False, fill_value=np.nan)
        lon_lr2d, lat_lr2d = np.meshgrid(lon_lr, lat_lr)
        egm_on_lr = interp(np.column_stack([lat_lr2d.ravel(), lon_lr2d.ravel()]))
        egm_on_lr = egm_on_lr.reshape(gravity_lr.shape)
        grav_lr_up = sp_zoom(egm_on_lr, (upscale, upscale), order=1)
        ship_grav[ship_mask] = ship_grav[ship_mask] + grav_lr_up[ship_mask]
        after_mean = np.nanmean(ship_grav[ship_mask])
        print(f"  Residual→FAA (EGM2008 trend): mean {before_mean:.0f} → {after_mean:.0f} mGal")
    except Exception as e:
        print(f"  EGM2008 unavailable ({e}), falling back to SSV trend")
        ssv_trend = gaussian_filter(gravity_lr, sigma=10)
        ssv_up = sp_zoom(ssv_trend, (upscale, upscale), order=1)
        ship_grav[ship_mask] = ship_grav[ship_mask] + ssv_up[ship_mask]
    return ship_grav


def butterworth_2d_filter(data, dx_deg, dy_deg, lat_center, cutoff_km, btype='low', order=4):
    """2D Butterworth 滤波器 (频域实现)。

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

    # Zero DC preserves mean for high-pass; low-pass keeps DC naturally
    if btype == 'high':
        H[0, 0] = 0.0

    data_fft = np.fft.fft2(data)
    filtered = np.real(np.fft.ifft2(data_fft * H))
    return filtered


def _estimate_coherence_cutoff(ship_filled, ssv_hr, dx_deg, dy_deg, lat_center):
    """估计相干函数 = 0.5 对应的截止波长 (ggad471 方法)。

    Returns cutoff_km (默认回退 27 km)。
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

        # Radial averaging in k-bins
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

        # Find first wavelength where coh drops to 0.5
        valid = lam > 0
        lam_v = lam[valid]
        coh_v = coh[valid]
        # Sort by wavelength ascending
        order = np.argsort(lam_v)
        lam_v = lam_v[order]
        coh_v = coh_v[order]
        idx = np.where(coh_v < 0.5)[0]
        if len(idx) > 0:
            cutoff_m = lam_v[idx[0]] * 1000  # km
            # Clamp to reasonable range [10, 60] km
            cutoff_m = float(np.clip(cutoff_m, 10, 60))
            print(f"  Estimated coherence=0.5 cutoff: {cutoff_m:.1f} km")
            return cutoff_m
    except Exception as e:
        print(f"  Coherence estimation failed ({e}), using default 27 km")
    return 27.0


def freq_fusion_reconstruction(lon_range=(140, 150), lat_range=(30, 40), upscale=4):
    """频域融合 重建 15" 重力场。"""
    print("=" * 60)
    print("  Baseline B: Frequency-domain Fusion (15'')")
    print("=" * 60)
    print(f"  Region: lon=[{lon_range[0]}, {lon_range[1]}], lat=[{lat_range[0]}, {lat_range[1]}]")

    print("\n[1/5] Loading SSV33.1 + shipborne data...")
    gravity_lr, _, lon_lr, lat_lr = load_ssv_region(lon_range, lat_range, load_vgg=False)
    ny_lr, nx_lr = gravity_lr.shape
    ny_hr, nx_hr = ny_lr * upscale, nx_lr * upscale
    lon_hr = np.linspace(lon_range[0], lon_range[1], nx_hr)
    lat_hr = np.linspace(lat_range[0], lat_range[1], ny_hr)
    print(f"  SSV 1' grid: {gravity_lr.shape}")
    print(f"  Target 15'' grid: ({ny_hr}, {nx_hr}) = {ny_hr*nx_hr/1e6:.1f}M pixels")

    ship_grav, _ = load_shipborne_gravity(lon_hr, lat_hr)
    ship_grav = _residual_to_faa(ship_grav, lon_range, lat_range, lon_lr, lat_lr, gravity_lr, upscale)
    n_ship = np.sum(~np.isnan(ship_grav))
    print(f"  Shipborne gravity points: {n_ship:,}")

    if n_ship < 100:
        print("  ERROR: insufficient shipborne data")
        return None

    print("\n[2/5] 80/20 shipborne split (seed=42)...")
    ship_train_mask, ship_val_mask = split_shipborne_mask(ship_grav, validation_fraction=0.2)
    print(f"  Train: {ship_train_mask.sum():,}  Val: {ship_val_mask.sum():,}")

    print("\n[3/5] Prepare aligned grids (SSV upsampled + shipborne SSV-filled)...")
    # SSV upsampled to 15" via bilinear
    ssv_hr = sp_zoom(gravity_lr, (upscale, upscale), order=1)

    # For training-only fusion: build a shipborne grid using ONLY training points
    ship_train_only = np.full_like(ship_grav, np.nan)
    ship_train_only[ship_train_mask] = ship_grav[ship_train_mask]

    # Fill NaN regions in ship_train_only with SSV before FFT
    # (ggad471 requires aligned grids; NaN regions get SSV continuum)
    nan_mask = np.isnan(ship_train_only)
    ship_filled = ship_train_only.copy()
    ship_filled[nan_mask] = ssv_hr[nan_mask]
    # Apply light Gaussian smoothing to suppress pixel-level noise from exact-pixel assignment
    ship_filled = gaussian_filter(ship_filled, sigma=2.0)
    print(f"  Ship grid NaN filled with SSV, coverage: {n_ship/ship_grav.size*100:.1f}%")

    # Smooth SSV for low-pass branch (remove sub-1' noise from bilinear upsampling)
    ssv_hr_smooth = gaussian_filter(ssv_hr, sigma=2.0)

    print("\n[4/5] Frequency-domain fusion (Butterworth, order=4)...")
    dx_deg = (lon_range[1] - lon_range[0]) / nx_hr
    dy_deg = (lat_range[1] - lat_range[0]) / ny_hr
    lat_center = (lat_range[0] + lat_range[1]) / 2.0

    # Estimate cutoff from coherence (fallback: 27 km per ggad471)
    cutoff_km = _estimate_coherence_cutoff(ship_filled, ssv_hr_smooth, dx_deg, dy_deg, lat_center)

    print(f"  Butterworth cutoff wavelength: {cutoff_km:.1f} km")
    print(f"  HP(ship) + LP(SSV) at order 4")

    t0 = time.time()
    ship_hp = butterworth_2d_filter(ship_filled, dx_deg, dy_deg, lat_center,
                                     cutoff_km, btype='high', order=4)
    ssv_lp = butterworth_2d_filter(ssv_hr_smooth, dx_deg, dy_deg, lat_center,
                                    cutoff_km, btype='low', order=4)
    gravity_fused = ssv_lp + ship_hp
    print(f"  FFT fusion done in {time.time()-t0:.1f}s")

    print("\n[5/5] Validation against held-out shipborne points...")
    gravity_bilinear = ssv_hr  # Bilinear = SSV upsampled

    m_fused = compute_metrics(gravity_fused, ship_grav, mask=ship_val_mask)
    m_bi = compute_metrics(gravity_bilinear, ship_grav, mask=ship_val_mask)

    print("\n" + "=" * 60)
    print("  RESULTS — Frequency-domain Fusion")
    print("=" * 60)
    print(f"  Fused vs Ship (val)   RMSE: {m_fused['RMSE']:.2f} mGal  "
          f"Corr: {m_fused['Correlation']:.4f}  Bias: {m_fused['Bias']:.2f}  "
          f"[n={m_fused['N']:,}]")
    print(f"  Bilinear vs Ship (val) RMSE: {m_bi['RMSE']:.2f} mGal  "
          f"Corr: {m_bi['Correlation']:.4f}")
    impr = (1 - m_fused['RMSE'] / m_bi['RMSE']) * 100 if m_bi['RMSE'] > 0 else 0
    print(f"  Fusion improvement over bilinear: {impr:+.1f}%")

    # Save
    np.save(OUTPUT_DIR / "gravity_freq_fused_15s.npy", gravity_fused)
    np.save(OUTPUT_DIR / "gravity_bilinear_15s.npy", gravity_bilinear)
    np.save(OUTPUT_DIR / "ship_val_mask_15s.npy", ship_val_mask)
    print(f"\n  Saved: {OUTPUT_DIR / 'gravity_freq_fused_15s.npy'}")

    return {
        'gravity_fused': gravity_fused,
        'gravity_bilinear': gravity_bilinear,
        'metrics_fused': m_fused,
        'metrics_bilinear': m_bi,
        'cutoff_km': cutoff_km,
        'ship_grav': ship_grav,
        'ship_val_mask': ship_val_mask,
        'lon_hr': lon_hr, 'lat_hr': lat_hr,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lon-min", type=float, default=140)
    parser.add_argument("--lon-max", type=float, default=150)
    parser.add_argument("--lat-min", type=float, default=30)
    parser.add_argument("--lat-max", type=float, default=40)
    parser.add_argument("--upscale", type=int, default=4)
    args = parser.parse_args()
    freq_fusion_reconstruction(
        lon_range=(args.lon_min, args.lon_max),
        lat_range=(args.lat_min, args.lat_max),
        upscale=args.upscale)
