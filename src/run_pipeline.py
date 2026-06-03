"""区域真实数据管线：加载 SSV+GEBCO+船载数据 → KF重力重建 → KF水深反演。"""
import sys, os, time, gzip, tempfile, re
from pathlib import Path

import numpy as np
import netCDF4 as nc
import xarray as xr
from scipy.ndimage import zoom, gaussian_filter
from scipy.interpolate import griddata
import matplotlib
matplotlib.use("Agg")  # 非交互式后端, 用于服务器/批处理生成图片
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from kalman_gravity import KalmanGravityReconstructor, KalmanBathymetryInverter
from utils import (compute_metrics, print_metrics, plot_gravity_field,
                   plot_bathymetry, plot_residual, plot_scatter_compare,
                   plot_error_histogram, interpolate_nans, bandpass_filter, compute_vgg,
                   split_shipborne_mask, read_gmt_grd, quality_control_shipborne)

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output_real"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NC_ENGINE = "h5netcdf"


def load_ssv_region(lon_range, lat_range, load_vgg=False):
    """加载指定区域的 SSV33.1 重力异常 (和可选的 VGG)。

    load_vgg=False 时只读 grav_33.1.nc, 节省 ~640MB IO。
    """
    grav_path = DATA_DIR / "ssv32" / "grav_33.1.nc"

    ds_g = xr.open_dataset(str(grav_path), engine=NC_ENGINE)
    if load_vgg:
        curv_path = DATA_DIR / "ssv32" / "curv_33.1.nc"
        ds_c = xr.open_dataset(str(curv_path), engine=NC_ENGINE)

    lat_slice = slice(lat_range[0], lat_range[1])
    ds_lon = ds_g.lon.values

    if ds_lon.max() > 180:
        lon0 = (lon_range[0] + 360) % 360
        lon1 = (lon_range[1] + 360) % 360
    else:
        lon0, lon1 = lon_range[0], lon_range[1]

    if lon0 < lon1:
        gravity = ds_g.z.sel(lon=slice(lon0, lon1), lat=lat_slice).values
        if load_vgg:
            vgg = ds_c.z.sel(lon=slice(lon0, lon1), lat=lat_slice).values
        lon = ds_g.lon.sel(lon=slice(lon0, lon1)).values
    else:
        lmax = 360 if ds_lon.max() > 180 else 180
        g1 = ds_g.z.sel(lon=slice(lon0, lmax), lat=lat_slice).values
        g2 = ds_g.z.sel(lon=slice(-lmax, lon1), lat=lat_slice).values
        gravity = np.concatenate([g1, g2], axis=1)
        if load_vgg:
            v1 = ds_c.z.sel(lon=slice(lon0, lmax), lat=lat_slice).values
            v2 = ds_c.z.sel(lon=slice(-lmax, lon1), lat=lat_slice).values
            vgg = np.concatenate([v1, v2], axis=1)
        l1 = ds_g.lon.sel(lon=slice(lon0, lmax)).values
        l2 = ds_g.lon.sel(lon=slice(-lmax, lon1)).values
        lon = np.sort(np.concatenate([l1, l2]))

    lat = ds_g.lat.sel(lat=lat_slice).values
    ds_g.close()
    if load_vgg:
        ds_c.close()
    return (gravity, vgg, lon, lat) if load_vgg else (gravity, None, lon, lat)


def load_gebco_region(lon_range, lat_range):
    """加载指定区域的 GEBCO sub-ice 海底地形。"""
    gebco_path = DATA_DIR / "gebco" / "GEBCO_2026_sub_ice" / "GEBCO_2026_sub_ice.nc"
    ds = xr.open_dataset(str(gebco_path), engine=NC_ENGINE)

    var_name = [v for v in ds.variables if len(ds[v].dims) >= 2][0]
    ds_var = ds[var_name]

    # Handle 0-360 vs -180-180 longitude conversion
    ds_lon = ds.lon.values
    lon0, lon1 = lon_range[0], lon_range[1]
    if lon0 < 0 and ds_lon.min() >= 0:
        lon0 += 360
        lon1 += 360

    pad = 0.5
    bathy = ds_var.sel(lon=slice(lon0 - pad, lon1 + pad),
                        lat=slice(lat_range[0] - pad, lat_range[1] + pad)).values
    lon_b = ds.lon.sel(lon=slice(lon0 - pad, lon1 + pad)).values
    lat_b = ds.lat.sel(lat=slice(lat_range[0] - pad, lat_range[1] + pad)).values
    ds.close()
    return bathy, lon_b, lat_b


def load_shipborne_gravity(lon_grid, lat_grid):
    """加载全部船载重力格网，格网化匹配目标经纬度。"""
    ship_dir = DATA_DIR / "shipborne"
    gravity_points = []
    bathy_points = []

    for cruise_dir in ship_dir.iterdir():
        if not cruise_dir.is_dir() or cruise_dir.name.startswith("_"):
            continue
        for f in list(cruise_dir.glob("*.grd.gz")) + list(cruise_dir.glob("*.grd")):
            fname = f.name.lower()
            is_gravity = any(kw in fname for kw in [
                "faa", "free_air", "free-air", "_gravity", "residual", "maba", "rmba", "bouguer",
                "ba."])
            is_bathy = any(kw in fname for kw in ["_mb", "bathy", "0.00025"])

            if not (is_gravity or is_bathy):
                continue

            try:
                data, lon, lat = read_gmt_grd(str(f))
                if data is None:
                    continue

                lon_2d, lat_2d = np.meshgrid(lon, lat)
                mask = ((lon_2d >= lon_grid.min()) & (lon_2d <= lon_grid.max()) &
                        (lat_2d >= lat_grid.min()) & (lat_2d <= lat_grid.max()))
                if mask.sum() == 0:
                    continue

                valid = mask & ~np.isnan(data)
                n_valid = valid.sum()
                # Subsample if too many points (e.g., high-res bathymetry)
                if n_valid > 500000:
                    step = max(1, n_valid // 500000)
                    idx_flat = np.where(valid.ravel())[0][::step]
                    idx = np.unravel_index(idx_flat, valid.shape)
                else:
                    idx = np.where(valid)
                n_pts = len(idx[0])
                pt_lon = lon[idx[1]]
                pt_lat = lat[idx[0]]
                pt_val = data[idx]
                if is_gravity:
                    grav_mask = np.abs(pt_val) < 5000
                    if grav_mask.sum() > 0:
                        gravity_points.append(np.column_stack([
                            pt_lon[grav_mask], pt_lat[grav_mask], pt_val[grav_mask]]))
                elif is_bathy:
                    bathy_mask = (pt_val > -12000) & (pt_val < 5000)
                    if bathy_mask.sum() > 0:
                        bathy_points.append(np.column_stack([
                            pt_lon[bathy_mask], pt_lat[bathy_mask], pt_val[bathy_mask]]))

                print(f"  Loaded {f.name}: {n_pts} points (from {n_valid:,})")
            except Exception as e:
                print(f"  Warning: skip {f.name}: {e}")

    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    ship_grav = np.full(lon_mesh.shape, np.nan)
    ship_bathy = np.full(lon_mesh.shape, np.nan)

    if gravity_points:
        pts = np.vstack(gravity_points)
        pts, unit = quality_control_shipborne(pts)
        ship_grav = griddata(pts[:, :2], pts[:, 2], (lon_mesh, lat_mesh), method="nearest")
    if bathy_points:
        pts = np.vstack(bathy_points)
        ship_bathy = griddata(pts[:, :2], pts[:, 2], (lon_mesh, lat_mesh), method="nearest")

    return ship_grav, ship_bathy


def run_pipeline(lon_range=(140, 150), lat_range=(30, 40), upscale=4):
    """完整管线：加载数据 → KF重力重建 → KF水深反演 → 评估+图表。"""
    print("=" * 60)
    print("  Kalman Filter Gravity Field Modeling - Real Data")
    print("=" * 60)
    print(f"  Region: lon=[{lon_range[0]}, {lon_range[1]}], lat=[{lat_range[0]}, {lat_range[1]}]")

    print("\n[1/3] Loading SSV33.1 satellite data...")
    gravity_lr, vgg_lr, lon_lr, lat_lr = load_ssv_region(lon_range, lat_range, load_vgg=True)
    print(f"  Gravity: {gravity_lr.shape}")
    if vgg_lr is not None:
        print(f"  VGG loaded: shape={vgg_lr.shape} (reserved for spectral validation)")

    print("\n[2/3] Loading GEBCO bathymetry...")
    bathy_gebco, lon_b, lat_b = load_gebco_region(lon_range, lat_range)
    print(f"  GEBCO: {bathy_gebco.shape}")

    print("\n[3/3] Loading shipborne data...")
    from scipy.ndimage import zoom as sp_zoom
    ny_lr, nx_lr = gravity_lr.shape
    ny_hr, nx_hr = ny_lr * upscale, nx_lr * upscale
    lon_hr = np.linspace(lon_range[0], lon_range[1], nx_hr)
    lat_hr = np.linspace(lat_range[0], lat_range[1], ny_hr)
    ship_grav, ship_bathy = load_shipborne_gravity(lon_hr, lat_hr)
    # Convert residual gravity → free-air: add back SSV regional trend
    if ship_grav is not None:
        ship_mask = ~np.isnan(ship_grav)
        if ship_mask.sum() > 0:
            grav_lr_up = sp_zoom(gravity_lr, (upscale, upscale), order=1)
            ship_grav[ship_mask] = ship_grav[ship_mask] + grav_lr_up[ship_mask]
    n_ship = np.sum(~np.isnan(ship_grav)) if ship_grav is not None else 0
    print(f"  Shipborne gravity points: {n_ship}")

    # Load GEBCO TID for independent validation
    print("\n[3b/6] Loading GEBCO TID type identifier...")
    prior_mask_hr = None
    val_mask_hr = None
    try:
        from real_data import load_gebco_tid, tid_to_multibeam_mask
        tid_info = load_gebco_tid(region=(lon_range[0], lon_range[1],
                                          lat_range[0], lat_range[1]))
        tid_data = tid_info["tid"]
        mb_mask = tid_to_multibeam_mask(tid_data)
        # Upsample TID mask to HR grid
        mb_mask_hr = sp_zoom(mb_mask.astype(float),
                             (ny_hr / mb_mask.shape[0], nx_hr / mb_mask.shape[1]),
                             order=0) > 0.5
        # 70/30 split for prior/validation within multibeam regions
        rng = np.random.default_rng(42)
        mb_idx = np.where(mb_mask_hr.ravel())[0]
        rng.shuffle(mb_idx)
        n_prior = int(len(mb_idx) * 0.7)
        prior_mask_hr = np.zeros(ny_hr * nx_hr, dtype=bool)
        val_mask_hr = np.zeros(ny_hr * nx_hr, dtype=bool)
        prior_mask_hr[mb_idx[:n_prior]] = True
        val_mask_hr[mb_idx[n_prior:]] = True
        prior_mask_hr = prior_mask_hr.reshape(ny_hr, nx_hr)
        val_mask_hr = val_mask_hr.reshape(ny_hr, nx_hr)
        print(f"  TID multibeam pixels: {len(mb_idx):,} "
              f"(prior={n_prior:,}, validation={len(mb_idx)-n_prior:,})")
    except (FileNotFoundError, KeyError) as e:
        print(f"  TID not available: {e} — GEBCO prior will be unrestricted")

    # Shipborne train/val split
    ship_train_mask = None
    ship_val_mask = None
    if n_ship > 100:
        ship_train_mask, ship_val_mask = split_shipborne_mask(
            ship_grav, validation_fraction=0.2)
        print(f"  Shipborne split: train={ship_train_mask.sum():,}, "
              f"val={ship_val_mask.sum():,}")

    print("\n[4/6] Kalman Filter Gravity Reconstruction...")
    reconstructor = KalmanGravityReconstructor(
        grid_shape=(ny_hr, nx_hr), region_size_deg=lat_range[1] - lat_range[0],
        lr_grid_shape=(ny_lr, nx_lr))
    reconstructor.initialize(
        gravity_lr=gravity_lr, gravity_ship=ship_grav if n_ship > 0 else None,
        ship_train_mask=ship_train_mask,
        process_noise_std=2.0, sat_noise_std=3.0, ship_noise_std=1.0)

    t0 = time.time()
    gravity_hr = reconstructor.run(n_iterations=6, smooth_sigma=0.8)
    print(f"  Done in {time.time()-t0:.0f}s")

    gravity_bilinear = zoom(gravity_lr, (upscale, upscale), order=1)

    # Spectral validation: band-separated metrics
    dx_km = (lat_range[1] - lat_range[0]) * 111.32 / ny_hr
    grav_kf_long = bandpass_filter(gravity_hr, dx_km, lam_min=160)
    grav_kf_mid = bandpass_filter(gravity_hr, dx_km, lam_min=20, lam_max=160)
    grav_bi_long = bandpass_filter(gravity_bilinear, dx_km, lam_min=160)
    grav_bi_mid = bandpass_filter(gravity_bilinear, dx_km, lam_min=20, lam_max=160)

    print("\n[4b/6] Spectral band validation...")
    print(f"  Long-wave (>160km) correlation KF vs bilinear: "
          f"{np.corrcoef(grav_kf_long.ravel(), grav_bi_long.ravel())[0,1]:.4f}")
    if ship_val_mask is not None and ship_val_mask.sum() > 100:
        rmse_kf_mid = compute_metrics(grav_kf_mid, gravity_hr, mask=ship_val_mask)
        print(f"  Mid-band (20-160km) energy fraction: "
              f"{np.std(grav_kf_mid[ship_val_mask])/np.std(gravity_hr[ship_val_mask]):.3f}")

    print("\n[5/6] Kalman Filter Bathymetry Inversion...")

    # --- VGG validation (Plan A) ---
    if vgg_lr is not None:
        from utils import compute_vgg
        vgg_kf = compute_vgg(gravity_hr, (lon_range[1] - lon_range[0]) / nx_hr,
                             lat_center=(lat_range[0] + lat_range[1]) / 2.0)
        vgg_lr_up = sp_zoom(vgg_lr, (upscale, upscale), order=1)
        m_vgg = compute_metrics(vgg_kf, vgg_lr_up)
        print(f"\n  [VGG Validation]")
        print(f"  VGG RMSE: {m_vgg.get('RMSE', float('nan')):.2f} Eötvös  |  "
              f"Corr: {m_vgg.get('Correlation', float('nan')):.3f}  |  "
              f"Bias: {m_vgg.get('Bias', float('nan')):.2f} Eötvös")
        # Report high-frequency difference specifically
        vgg_diff = vgg_kf - vgg_lr_up
        print(f"  VGG diff std: {np.nanstd(vgg_diff):.2f} Eötvös  "
              f"(large → KF needs VGG constraint)")
        if ship_val_mask is not None and ship_val_mask.sum() > 100:
            ratio = np.nanstd(vgg_diff[ship_val_mask]) / np.nanstd(vgg_lr_up[ship_val_mask])
            print(f"  VGG diff/signal ratio (ship val area): {ratio:.3f}  "
                  f"({'HIGH — consider adding VGG constraint' if ratio > 0.5 else 'LOW — VGG constraint unnecessary'})")

    inverter = KalmanBathymetryInverter(
        grid_shape=(ny_hr, nx_hr), region_size_deg=lat_range[1] - lat_range[0],
        lat_center=(lat_range[0] + lat_range[1]) / 2.0)

    bathy_prior_hr = sp_zoom(bathy_gebco, (ny_hr / bathy_gebco.shape[0],
                                            nx_hr / bathy_gebco.shape[1]), order=1)
    inverter.depth_min = max(-11000, np.nanmin(bathy_prior_hr) - 500)
    inverter.depth_max = min(1000, np.nanmax(bathy_prior_hr) + 500)

    # Use data-driven Te from crustal age if available
    try:
        from real_data import load_age_grid
        age_info = load_age_grid(region=(lon_range[0], lon_range[1],
                                          lat_range[0], lat_range[1]))
        inverter.apply_flexural_isostasy_from_age(age_info["age"])
    except (FileNotFoundError, KeyError) as e:
        print(f"  Age grid not available: {e} — using default Te=10 km")
        inverter.apply_flexural_isostasy(Te=10e3)

    inverter.initialize(gravity_reconstructed=gravity_hr,
                        gravity_ship=ship_grav if n_ship > 0 else None,
                        bathy_ship=ship_bathy, bathymetry_prior=bathy_prior_hr,
                        prior_mask=prior_mask_hr,
                        process_noise_std=80.0, observation_noise_std=5.0)

    # Apply GlobSed sediment correction (must be after initialize)
    try:
        from real_data import load_globsed
        globsed_info = load_globsed(region=(lon_range[0], lon_range[1],
                                             lat_range[0], lat_range[1]))
        inverter.apply_sediment_correction(globsed_info["thickness"], drho_sed=500)
    except (FileNotFoundError, KeyError) as e:
        print(f"  GlobSed not available: {e}")

    bathymetry_kf = inverter.run(n_iterations=5, smooth_sigma=1.0)
    bathymetry_trad = inverter.gravity_to_bathymetry_linear(
        gravity_hr, k_min=2 * np.pi / 160000, k_max=2 * np.pi / 20000)

    bathy_truth_hr = sp_zoom(bathy_gebco, (ny_hr / bathy_gebco.shape[0],
                                            nx_hr / bathy_gebco.shape[1]), order=1)
    m_g_kf = compute_metrics(gravity_hr, gravity_bilinear)
    # Independent validation on TID=1 held-out pixels
    if val_mask_hr is not None and val_mask_hr.sum() > 100:
        m_b_kf = compute_metrics(bathymetry_kf, bathy_truth_hr, mask=val_mask_hr)
        m_b_trad = compute_metrics(bathymetry_trad, bathy_truth_hr, mask=val_mask_hr)
        val_tag = f"[TID=1 独立验证, n={val_mask_hr.sum():,}]"
    else:
        m_b_kf = compute_metrics(bathymetry_kf, bathy_truth_hr)
        m_b_trad = compute_metrics(bathymetry_trad, bathy_truth_hr)
        val_tag = "[所有 GEBCO 像素 — WARNING: 先验和验证同源, 结果非独立!]"

    # Gravity validation on held-out shipborne
    if ship_val_mask is not None and ship_val_mask.sum() > 100:
        m_g_ship_val = compute_metrics(
            gravity_hr, ship_grav, mask=ship_val_mask)
    else:
        m_g_ship_val = None

    # Independent bathy validation on shipborne (never used in KF)
    n_bathy_ship = np.sum(~np.isnan(ship_bathy)) if ship_bathy is not None else 0
    m_b_kf_ship = None
    m_b_trad_ship = None
    if n_bathy_ship > 100:
        mask_b = ~np.isnan(ship_bathy)
        m_b_kf_ship = compute_metrics(bathymetry_kf[mask_b], ship_bathy[mask_b])
        m_b_trad_ship = compute_metrics(bathymetry_trad[mask_b], ship_bathy[mask_b])

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Gravity KF mean/std: {gravity_hr.mean():.1f} / {gravity_hr.std():.1f} mGal")
    if m_g_ship_val:
        print(f"  Gravity KF vs Ship (验证) RMSE: {m_g_ship_val['RMSE']:.1f} mGal "
              f"[n={m_g_ship_val.get('N', '?')}]")
    print(f"  Bathymetry KF mean: {bathymetry_kf.mean():.0f} m")
    print(f"  Bathymetry RMSE {val_tag}: {m_b_kf['RMSE']:.0f} m (KF) vs {m_b_trad['RMSE']:.0f} m (Parker)")
    if m_b_kf_ship:
        print(f"  Bathymetry RMSE vs 船载(独立): {m_b_kf_ship['RMSE']:.0f} m (KF) vs {m_b_trad_ship['RMSE']:.0f} m (Parker)")

    plot_gravity_field(
        [gravity_lr, gravity_hr, gravity_bilinear],
        ["SSV33.1 (1')", "卡尔曼滤波 (15'')", "双线性插值 (15'')"],
        save_path=OUTPUT_DIR / "real_gravity.png")
    plot_bathymetry(
        [bathy_prior_hr, bathymetry_kf, bathy_truth_hr],
        ["GEBCO 先验", "卡尔曼滤波", "GEBCO 真值"],
        save_path=OUTPUT_DIR / "real_bathymetry.png")
    plot_residual(bathymetry_kf, bathy_truth_hr,
                  title=f"水深残差 (RMSE={m_b_kf['RMSE']:.0f}m)",
                  save_path=OUTPUT_DIR / "real_residual.png")

    print(f"\nOutput saved to: {OUTPUT_DIR}")
    return gravity_hr, bathymetry_kf


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lon-min", type=float, default=140)
    parser.add_argument("--lon-max", type=float, default=150)
    parser.add_argument("--lat-min", type=float, default=30)
    parser.add_argument("--lat-max", type=float, default=40)
    parser.add_argument("--upscale", type=int, default=4)
    args = parser.parse_args()

    run_pipeline(lon_range=(args.lon_min, args.lon_max),
                 lat_range=(args.lat_min, args.lat_max), upscale=args.upscale)
