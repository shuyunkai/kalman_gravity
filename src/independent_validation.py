"""独立巡航验证: 用 MR22-04 作为完全独立的验证集, 其余 30 个巡航训练.

验证机制:
  - 训练: 30 个巡航的所有原始点 → 精确像素赋值到 15" 网格
  - 验证: MR22-04 的原始 10 秒采样点 (从未参与训练)
  - 在每个验证点上, 从 15" 重建网格双线性插值取值, 与该点 FAA 对比

对比方法:
  1. Bilinear (SSV 1' → 15'')
  2. KF 原版 (15'' 网格)
  3. KF v2b (1' 网格 + 双线性上采样)
  4. Kriging + 趋势面 (1' 网格 + 双线性上采样)
"""
import sys, time, re
from pathlib import Path
import numpy as np
from scipy.ndimage import zoom as sp_zoom, gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))

from run_pipeline import load_ssv_region, NC_ENGINE, DATA_DIR
from kriging_baseline import _residual_to_faa, _subsample_train_points
from kalman_gravity import KalmanGravityReconstructor
from utils import compute_metrics

JP_DATA_DIR = Path(__file__).parent.parent / "jp_data"
OUTPUT_DIR = Path(__file__).parent.parent / "output_real"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 独立验证巡航
VAL_CRUISE = "MR22-04_gravity_dmo"


def load_cruise_points(cruise_dir, lon_range, lat_range):
    """加载一个巡航的 .grv 文件, 返回区域内的 (lon, lat, faa) 数组。

    .grv 格式: 日期 时间 纬度 经度 Eötvös校正 FAA(mGal)
    """
    grv_files = list(cruise_dir.glob("*.grv"))
    if not grv_files:
        return None
    f = grv_files[0]
    pts = []
    with open(f, 'r', errors='ignore') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                lat = float(parts[2])
                lon = float(parts[3])
                faa = float(parts[5])
            except (ValueError, IndexError):
                continue
            if (lon_range[0] <= lon <= lon_range[1] and
                lat_range[0] <= lat <= lat_range[1] and
                abs(faa) < 5000):
                pts.append([lon, lat, faa])
    if not pts:
        return None
    return np.array(pts, dtype=np.float64)


def load_all_cruises_split(lon_range, lat_range, val_cruise=VAL_CRUISE):
    """加载所有巡航, 分为训练集和验证集 (按巡航划分)."""
    train_points = []
    val_points = None
    val_cruise_dir = JP_DATA_DIR / val_cruise
    print(f"  Validation cruise: {val_cruise}")
    n_train_cruises = 0
    for cruise_dir in sorted(JP_DATA_DIR.iterdir()):
        if not cruise_dir.is_dir():
            continue
        pts = load_cruise_points(cruise_dir, lon_range, lat_range)
        if pts is None or len(pts) == 0:
            continue
        if cruise_dir.name == val_cruise:
            val_points = pts
            print(f"    [VAL] {cruise_dir.name}: {len(pts):,} points")
        else:
            train_points.append(pts)
            n_train_cruises += 1
            print(f"    [train] {cruise_dir.name}: {len(pts):,} points")
    if val_points is None:
        raise FileNotFoundError(f"Validation cruise {val_cruise} not found or empty")
    train_all = np.vstack(train_points)
    print(f"  Total: {n_train_cruises} train cruises, {len(train_all):,} train points")
    print(f"         1 val cruise ({val_cruise}), {len(val_points):,} val points")
    return train_all, val_points


def assign_points_to_grid(points, lon_grid, lat_grid):
    """精确像素赋值: (N,3) points → 2D grid (last write wins)."""
    ny, nx = len(lat_grid), len(lon_grid)
    grid = np.full((ny, nx), np.nan)
    j_idx = np.round((points[:, 0] - lon_grid[0]) / (lon_grid[1] - lon_grid[0])).astype(int)
    i_idx = np.round((points[:, 1] - lat_grid[0]) / (lat_grid[1] - lat_grid[0])).astype(int)
    valid = ((j_idx >= 0) & (j_idx < nx) &
             (i_idx >= 0) & (i_idx < ny))
    grid[i_idx[valid], j_idx[valid]] = points[valid, 2]
    return grid


def downsample_to_1min(ship_15s, hr_per_lr=4):
    """15" 训练网格 → 1' 训练网格 (每 4×4 块训练点均值)."""
    ny_hr, nx_hr = ship_15s.shape
    ny_lr = ny_hr // hr_per_lr
    nx_lr = nx_hr // hr_per_lr
    ship_1m = np.full((ny_lr, nx_lr), np.nan)
    train_1m = np.zeros((ny_lr, nx_lr), dtype=bool)
    for i in range(ny_lr):
        for j in range(nx_lr):
            block = ship_15s[i*hr_per_lr:(i+1)*hr_per_lr,
                               j*hr_per_lr:(j+1)*hr_per_lr]
            valid = ~np.isnan(block)
            if np.any(valid):
                ship_1m[i, j] = np.nanmean(block)
                train_1m[i, j] = True
    return ship_1m, train_1m


def evaluate_on_val_points(gravity_15s, val_points, lon_hr, lat_hr):
    """在原始验证点上评估: 双线性插值从 15" 网格取值, 与验证点 FAA 对比."""
    interp = RegularGridInterpolator(
        (lat_hr, lon_hr), gravity_15s,
        bounds_error=False, fill_value=np.nan)
    pred = interp(np.column_stack([val_points[:, 1], val_points[:, 0]]))
    truth = val_points[:, 2]
    valid = ~(np.isnan(pred) | np.isnan(truth))
    pred_v = pred[valid]
    truth_v = truth[valid]
    diff = pred_v - truth_v
    n = len(diff)
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    corr = float(np.corrcoef(pred_v, truth_v)[0, 1]) if n > 1 else 0.0
    return {'RMSE': rmse, 'MAE': mae, 'Bias': bias, 'Correlation': corr, 'N': n}


def run_kf_15s(gravity_lr, ship_grav, train_mask, lon_range, lat_range, upscale=4,
               process_noise_std=2.0, sat_noise_std=3.0, ship_noise_std=1.0,
               n_iterations=6, smooth_sigma=15.0):
    """KF 原版: 15" 网格."""
    ny_lr, nx_lr = gravity_lr.shape
    ny_hr, nx_hr = ny_lr * upscale, nx_lr * upscale
    r = KalmanGravityReconstructor(
        grid_shape=(ny_hr, nx_hr),
        region_size_deg=lat_range[1] - lat_range[0],
        lr_grid_shape=(ny_lr, nx_lr))
    r.initialize(
        gravity_lr=gravity_lr, gravity_ship=ship_grav,
        ship_train_mask=train_mask,
        process_noise_std=process_noise_std,
        sat_noise_std=sat_noise_std,
        ship_noise_std=ship_noise_std)
    return r.run(n_iterations=n_iterations, smooth_sigma=smooth_sigma)


def run_kf_1min(gravity_lr, ship_grav, train_mask, lon_range, lat_range, upscale=4,
                process_noise_std=2.0, sat_noise_std=3.0, ship_noise_std=1.0,
                n_iterations=6, smooth_sigma=4.0):
    """KF v2b: 1' 网格 + 双线性上采样."""
    ny_lr, nx_lr = gravity_lr.shape
    ship_1m, train_1m = downsample_to_1min(ship_grav, hr_per_lr=upscale)
    r = KalmanGravityReconstructor(
        grid_shape=(ny_lr, nx_lr),
        region_size_deg=lat_range[1] - lat_range[0],
        lr_grid_shape=(ny_lr, nx_lr))
    r.initialize(
        gravity_lr=gravity_lr, gravity_ship=ship_1m,
        ship_train_mask=train_1m,
        process_noise_std=process_noise_std,
        sat_noise_std=sat_noise_std,
        ship_noise_std=ship_noise_std)
    grav_1m = r.run(n_iterations=n_iterations, smooth_sigma=smooth_sigma)
    return sp_zoom(grav_1m, (upscale, upscale), order=1)


def run_kriging(gravity_lr, ship_grav, train_mask, lon_hr, lat_hr,
                lon_range, lat_range, upscale=4, n_target=3000):
    """克里金+趋势面: 1' 网格 + 双线性上采样."""
    from pykrige.ok import OrdinaryKriging
    ny_lr, nx_lr = gravity_lr.shape
    lon_lr = np.linspace(lon_range[0], lon_range[1], nx_lr)
    lat_lr = np.linspace(lat_range[0], lat_range[1], ny_lr)

    lon_pts, lat_pts, val_pts = _subsample_train_points(
        lon_hr, lat_hr, ship_grav, train_mask, n_target=n_target, seed=42)
    lon0, lat0 = lon_range[0], lat_range[0]
    dlat = lat_pts - lat0
    dlon = lon_pts - lon0
    A_train = np.column_stack([np.ones_like(dlat), dlat, dlon, dlat*dlon, dlat**2, dlon**2])
    coef, *_ = np.linalg.lstsq(A_train, val_pts, rcond=None)
    resid_train = val_pts - A_train @ coef
    resid_var = np.var(resid_train)
    nugget = 5.0
    sill = max(resid_var, nugget + 1.0)
    rng_a = 0.25
    OK = OrdinaryKriging(
        lon_pts, lat_pts, resid_train,
        variogram_model='spherical',
        variogram_parameters={'nugget': nugget, 'range': rng_a, 'sill': sill},
        verbose=False, enable_plotting=False,
        exact_values=False, pseudo_inv=True)
    z_krig_1m, _ = OK.execute('grid', lon_lr, lat_lr, backend='vectorized')
    z_krig_1m = np.asarray(z_krig_1m)
    if z_krig_1m.shape != (ny_lr, nx_lr):
        z_krig_1m = z_krig_1m.T
    ii_1m, jj_1m = np.meshgrid(np.arange(ny_lr), np.arange(nx_lr), indexing='ij')
    dlat_g1 = lat_lr[ii_1m] - lat0
    dlon_g1 = lon_lr[jj_1m] - lon0
    A_grid_1m = np.column_stack([np.ones(ny_lr * nx_lr), dlat_g1.ravel(), dlon_g1.ravel(),
                                  (dlat_g1 * dlon_g1).ravel(),
                                  (dlat_g1**2).ravel(), (dlon_g1**2).ravel()])
    trend_1m = (A_grid_1m @ coef).reshape(ny_lr, nx_lr)
    gravity_krig_1m = trend_1m + z_krig_1m
    return sp_zoom(gravity_krig_1m, (upscale, upscale), order=1)


def main(lon_range=(140, 150), lat_range=(30, 40), upscale=4):
    print("=" * 70)
    print(f"  Independent Cruise Validation (val={VAL_CRUISE})")
    print("=" * 70)
    print(f"  Region: lon=[{lon_range[0]}, {lon_range[1]}], lat=[{lat_range[0]}, {lat_range[1]}]")

    print("\n[1/5] Loading SSV33.1...")
    gravity_lr, _, lon_lr, lat_lr = load_ssv_region(lon_range, lat_range, load_vgg=False)
    ny_lr, nx_lr = gravity_lr.shape
    ny_hr, nx_hr = ny_lr * upscale, nx_lr * upscale
    lon_hr = np.linspace(lon_range[0], lon_range[1], nx_hr)
    lat_hr = np.linspace(lat_range[0], lat_range[1], ny_hr)
    print(f"  SSV 1': {gravity_lr.shape}  Target 15'': ({ny_hr}, {nx_hr})")

    print("\n[2/5] Loading 31 cruises (30 train + 1 val)...")
    train_points, val_points = load_all_cruises_split(lon_range, lat_range, VAL_CRUISE)

    print("\n[3/5] Assigning train points to 15'' grid (exact pixel)...")
    ship_grav = assign_points_to_grid(train_points, lon_hr, lat_hr)
    n_ship = np.sum(~np.isnan(ship_grav))
    print(f"  Train grid non-NaN pixels: {n_ship:,} ({n_ship/ship_grav.size*100:.1f}% coverage)")

    # Residual → FAA (same as run_pipeline)
    ship_grav = _residual_to_faa(ship_grav, lon_range, lat_range, lon_lr, lat_lr, gravity_lr, upscale)

    # Training mask = all non-NaN pixels (no random split — validation is fully independent)
    train_mask = ~np.isnan(ship_grav)
    print(f"  Train mask: {train_mask.sum():,} pixels")

    # Also convert val points to FAA if they are residual (mean check)
    val_mean = np.mean(val_points[:, 2])
    print(f"  Val points FAA mean: {val_mean:.1f} mGal  "
          f"(if |mean|<50 → residual, need trend recovery)")
    if abs(val_mean) < 50:
        print("  WARNING: val points appear to be residual, applying same EGM2008 trend")
        # Build EGM2008 trend on val points
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
            trend_at_val = interp(np.column_stack([val_points[:, 1], val_points[:, 0]]))
            val_points[:, 2] = val_points[:, 2] + trend_at_val
            print(f"  Val points after trend recovery: mean={np.mean(val_points[:, 2]):.1f} mGal")
        except Exception as e:
            print(f"  EGM2008 trend for val points failed: {e}")

    print("\n[4/5] Reconstructing gravity with 4 methods...")
    results = {}

    # Method 1: Bilinear
    print("\n  [a] Bilinear (SSV 1' → 15'')...")
    gravity_bilinear = sp_zoom(gravity_lr, (upscale, upscale), order=1)
    m_bi = evaluate_on_val_points(gravity_bilinear, val_points, lon_hr, lat_hr)
    print(f"      RMSE={m_bi['RMSE']:.2f} mGal  Corr={m_bi['Correlation']:.4f}  N={m_bi['N']:,}")
    results['Bilinear'] = {'gravity': gravity_bilinear, 'metrics': m_bi}

    # Method 2: KF original (15'' grid)
    print("\n  [b] KF original (15'' grid, smooth_sigma=15)...")
    t0 = time.time()
    gravity_kf = run_kf_15s(gravity_lr, ship_grav, train_mask, lon_range, lat_range, upscale)
    m_kf = evaluate_on_val_points(gravity_kf, val_points, lon_hr, lat_hr)
    print(f"      done {time.time()-t0:.0f}s | RMSE={m_kf['RMSE']:.2f}  Corr={m_kf['Correlation']:.4f}")
    results['KF_orig'] = {'gravity': gravity_kf, 'metrics': m_kf}

    # Method 3: KF v2b (1' grid + bilinear)
    print("\n  [c] KF v2b (1' grid + bilinear upsample)...")
    t0 = time.time()
    gravity_kf_v2b = run_kf_1min(gravity_lr, ship_grav, train_mask, lon_range, lat_range, upscale)
    m_kf_v2b = evaluate_on_val_points(gravity_kf_v2b, val_points, lon_hr, lat_hr)
    print(f"      done {time.time()-t0:.0f}s | RMSE={m_kf_v2b['RMSE']:.2f}  Corr={m_kf_v2b['Correlation']:.4f}")
    results['KF_v2b'] = {'gravity': gravity_kf_v2b, 'metrics': m_kf_v2b}

    # Method 4: Kriging + Trend
    print("\n  [d] Kriging+Trend (1' grid + bilinear upsample)...")
    t0 = time.time()
    gravity_krig = run_kriging(gravity_lr, ship_grav, train_mask, lon_hr, lat_hr,
                                 lon_range, lat_range, upscale)
    m_krig = evaluate_on_val_points(gravity_krig, val_points, lon_hr, lat_hr)
    print(f"      done {time.time()-t0:.0f}s | RMSE={m_krig['RMSE']:.2f}  Corr={m_krig['Correlation']:.4f}")
    results['Kriging'] = {'gravity': gravity_krig, 'metrics': m_krig}

    # === Summary ===
    print("\n" + "=" * 70)
    print(f"  INDEPENDENT CRUISE VALIDATION SUMMARY  (val={VAL_CRUISE}, n={m_bi['N']:,})")
    print("=" * 70)
    print(f"  {'Method':<22} {'RMSE':<10} {'MAE':<10} {'Corr':<8} {'Bias':<10} {'Impr%':<8}")
    print("  " + "-" * 68)
    base_rmse = m_bi['RMSE']
    for name, r in results.items():
        m = r['metrics']
        impr = (1 - m['RMSE'] / base_rmse) * 100
        print(f"  {name:<22} {m['RMSE']:<10.2f} {m['MAE']:<10.2f} "
              f"{m['Correlation']:<8.4f} {m['Bias']:<10.2f} {impr:<+8.2f}")

    # Save
    for name, r in results.items():
        np.save(OUTPUT_DIR / f"indepval_{name}.npy", r['gravity'])
    np.save(OUTPUT_DIR / "indepval_val_points.npy", val_points)
    print(f"\n  Outputs saved to {OUTPUT_DIR}")

    # Summary text
    lines = [
        "=" * 70,
        f"  Independent Cruise Validation (val={VAL_CRUISE})",
        "=" * 70,
        f"  Region: lon=[{lon_range[0]}, {lon_range[1]}], lat=[{lat_range[0]}, {lat_range[1]}]",
        f"  Train: 30 cruises, {n_ship:,} 15'' grid pixels",
        f"  Val:   {VAL_CRUISE}, {m_bi['N']:,} original 10s points (fully independent)",
        "",
        f"  {'Method':<22} {'RMSE':<10} {'MAE':<10} {'Corr':<8} {'Bias':<10} {'Impr%':<8}",
        "  " + "-" * 68,
    ]
    for name, r in results.items():
        m = r['metrics']
        impr = (1 - m['RMSE'] / base_rmse) * 100
        lines.append(f"  {name:<22} {m['RMSE']:<10.2f} {m['MAE']:<10.2f} "
                     f"{m['Correlation']:<8.4f} {m['Bias']:<10.2f} {impr:<+8.2f}")
    summary_text = "\n".join(lines)
    print("\n" + summary_text)
    with open(OUTPUT_DIR / "independent_validation_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lon-min", type=float, default=140)
    parser.add_argument("--lon-max", type=float, default=150)
    parser.add_argument("--lat-min", type=float, default=30)
    parser.add_argument("--lat-max", type=float, default=40)
    parser.add_argument("--upscale", type=int, default=4)
    args = parser.parse_args()
    main(
        lon_range=(args.lon_min, args.lon_max),
        lat_range=(args.lat_min, args.lat_max),
        upscale=args.upscale)
