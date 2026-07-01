"""基线方法 A: 普通克里金 + 趋势面 重建 15" 重力场。

参考论文:
  - ggad471 (Chen et al. 2024, GJI): Method 3 — 多面函数 + 残差约束趋势面
  - essd-16-4119-2024 (Li et al. 2024, ESSD): LSC 协方差模型 (数学等价克里金)

数学模型:
    Δg(φ,λ) = T(φ,λ) + R(φ,λ)
    T = a0 + a1·Δφ + a2·Δλ + a3·ΔφΔλ + a4·Δφ² + a5·Δλ²   (二次多项式趋势面)
    R = Σ λᵢ rᵢ                                          (普通克里金残差)
    λ 由变差函数 γ(h) 求解: 球状模型 γ(h) = C0 + C(3h/2a - h³/2a³)

与本项目 KF 在相同 15" 网格、相同 80/20 船载划分上对比。
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
    """复用 run_pipeline 的 EGM2008 趋势恢复逻辑, 把残余重力 → FAA。"""
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


def _subsample_train_points(lon_hr, lat_hr, ship_grav, train_mask, n_target=6000, seed=42):
    """空间分层抽样: 把训练点降到 ~6000, 让克里金矩阵可解 (6000² ≈ 36M 元素)。

    每 0.1°×0.1° 单元内随机取 1 个代表点, 不足再用全局随机补足。
    """
    rng = np.random.default_rng(seed)
    train_idx = np.where(train_mask.ravel())[0]
    if len(train_idx) <= n_target:
        lon_pts = lon_hr[np.unravel_index(train_idx, ship_grav.shape)[1]]
        lat_pts = lat_hr[np.unravel_index(train_idx, ship_grav.shape)[0]]
        val_pts = ship_grav.ravel()[train_idx]
        return lon_pts, lat_pts, val_pts

    cell_deg = 0.1
    lon_grid = np.floor(ship_grav.shape[1] * 0 + (np.unravel_index(train_idx, ship_grav.shape)[1].astype(float) * (lon_hr[1] - lon_hr[0]) / cell_deg)) * 0
    # Simpler: build cell index from lat/lon directly
    ii_full, jj_full = np.unravel_index(train_idx, ship_grav.shape)
    lat_pts_all = lat_hr[ii_full]
    lon_pts_all = lon_hr[jj_full]
    val_pts_all = ship_grav.ravel()[train_idx]

    cell_i = np.floor((lat_pts_all - lat_hr[0]) / cell_deg).astype(int)
    cell_j = np.floor((lon_pts_all - lon_hr[0]) / cell_deg).astype(int)
    cell_key = cell_i.astype(np.int64) * 100000 + cell_j

    # Group by cell, pick one random per cell
    unique_cells, first_idx = np.unique(cell_key, return_index=True)
    rng.shuffle(first_idx)
    picked = list(first_idx)

    # If not enough, top up with random from remaining
    if len(picked) < n_target:
        remaining = np.setdiff1d(np.arange(len(train_idx)), np.array(picked), assume_unique=False)
        extra = rng.choice(remaining, size=min(n_target - len(picked), len(remaining)), replace=False)
        picked.extend(extra.tolist())
    else:
        picked = picked[:n_target]

    picked = np.array(picked)
    return lon_pts_all[picked], lat_pts_all[picked], val_pts_all[picked]


def kriging_reconstruction(lon_range=(140, 150), lat_range=(30, 40), upscale=4):
    """普通克里金 + 趋势面 重建 15" 重力场。"""
    print("=" * 60)
    print("  Baseline A: Ordinary Kriging + Trend Surface (15'')")
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

    print("\n[3/5] Trend surface fit (6-term quadratic polynomial)...")
    lon_pts, lat_pts, val_pts = _subsample_train_points(
        lon_hr, lat_hr, ship_grav, ship_train_mask, n_target=6000, seed=42)
    print(f"  Subsampled train points: {len(lon_pts):,}")

    # Design matrix: [1, dlat, dlon, dlat*dlon, dlat², dlon²]
    lat0 = lat_range[0]
    lon0 = lon_range[0]
    dlat = lat_pts - lat0
    dlon = lon_pts - lon0
    A_train = np.column_stack([np.ones_like(dlat), dlat, dlon, dlat * dlon, dlat**2, dlon**2])
    coef, *_ = np.linalg.lstsq(A_train, val_pts, rcond=None)
    trend_train = A_train @ coef
    resid_train = val_pts - trend_train
    resid_var = np.var(resid_train)
    print(f"  Trend coefficients: {coef}")
    print(f"  Residual std after detrend: {np.std(resid_train):.2f} mGal")

    print("\n[4/5] Ordinary Kriging on residuals (spherical variogram)...")
    try:
        from pykrige.ok import OrdinaryKriging
    except ImportError:
        print("  pykrige not installed. Run: pip install pykrige")
        return None

    # Spherical variogram parameters
    # nugget C0: cross-track disagreement variance (essd-16-4119: ~8 mGal²)
    # range a: 3-5x ship point spacing (5-10 km → a=25 km)
    # sill C0+C: residual variance
    nugget = 5.0
    sill = max(resid_var, nugget + 1.0)
    rng_a = 0.25  # degrees ≈ 27 km at 35°N
    variogram_parameters = {'nugget': nugget, 'range': rng_a, 'sill': sill}
    print(f"  Variogram (spherical): nugget={nugget:.1f}, range={rng_a}°, sill={sill:.1f}")

    t0 = time.time()
    OK = OrdinaryKriging(
        lon_pts, lat_pts, resid_train,
        variogram_model='spherical',
        variogram_parameters=variogram_parameters,
        verbose=False, enable_plotting=False,
        exact_values=False, pseudo_inv=True)
    print(f"  Kriging matrix built in {time.time()-t0:.1f}s")

    # Execute on the HR grid (pykrige returns (ny, nx) with y ascending)
    t0 = time.time()
    # Use 'grid' for batch grid interpolation; backbone returns (n_lat, n_lon)
    z_krig, _ = OK.execute('grid', lon_hr, lat_hr, backend='vectorized')
    print(f"  Kriging grid execution in {time.time()-t0:.1f}s")
    z_krig = np.asarray(z_krig)
    if z_krig.shape != (ny_hr, nx_hr):
        z_krig = z_krig.T

    # Reconstruct full gravity: trend + kriged residual
    ii_all, jj_all = np.meshgrid(np.arange(ny_hr), np.arange(nx_hr), indexing='ij')
    dlat_grid = lat_hr[ii_all] - lat0
    dlon_grid = lon_hr[jj_all] - lon0
    A_grid = np.column_stack([np.ones(ny_hr * nx_hr), dlat_grid.ravel(), dlon_grid.ravel(),
                              (dlat_grid * dlon_grid).ravel(),
                              (dlat_grid**2).ravel(), (dlon_grid**2).ravel()])
    trend_grid = (A_grid @ coef).reshape(ny_hr, nx_hr)
    gravity_krig = trend_grid + z_krig

    print("\n[5/5] Validation against held-out shipborne points...")
    gravity_bilinear = sp_zoom(gravity_lr, (upscale, upscale), order=1)

    m_krig = compute_metrics(gravity_krig, ship_grav, mask=ship_val_mask)
    m_bi = compute_metrics(gravity_bilinear, ship_grav, mask=ship_val_mask)

    print("\n" + "=" * 60)
    print("  RESULTS — Ordinary Kriging + Trend")
    print("=" * 60)
    print(f"  Kriging vs Ship (val)  RMSE: {m_krig['RMSE']:.2f} mGal  "
          f"Corr: {m_krig['Correlation']:.4f}  Bias: {m_krig['Bias']:.2f}  "
          f"[n={m_krig['N']:,}]")
    print(f"  Bilinear vs Ship (val) RMSE: {m_bi['RMSE']:.2f} mGal  "
          f"Corr: {m_bi['Correlation']:.4f}")
    impr = (1 - m_krig['RMSE'] / m_bi['RMSE']) * 100 if m_bi['RMSE'] > 0 else 0
    print(f"  Kriging improvement over bilinear: {impr:+.1f}%")

    # Save result
    np.save(OUTPUT_DIR / "gravity_kriging_15s.npy", gravity_krig)
    np.save(OUTPUT_DIR / "gravity_bilinear_15s.npy", gravity_bilinear)
    np.save(OUTPUT_DIR / "ship_val_mask_15s.npy", ship_val_mask)
    print(f"\n  Saved: {OUTPUT_DIR / 'gravity_kriging_15s.npy'}")

    return {
        'gravity_krig': gravity_krig,
        'gravity_bilinear': gravity_bilinear,
        'metrics_krig': m_krig,
        'metrics_bilinear': m_bi,
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
    kriging_reconstruction(
        lon_range=(args.lon_min, args.lon_max),
        lat_range=(args.lat_min, args.lat_max),
        upscale=args.upscale)
