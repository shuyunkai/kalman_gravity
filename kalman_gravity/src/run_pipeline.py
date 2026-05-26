"""区域真实数据管线：加载 SSV+GEBCO+船载数据 → KF重力重建 → KF水深反演。"""
import sys, os, time, gzip, tempfile, re
from pathlib import Path

import numpy as np
import netCDF4 as nc
import xarray as xr
from scipy.ndimage import zoom, gaussian_filter
from scipy.interpolate import griddata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from kalman_gravity import KalmanGravityReconstructor, KalmanBathymetryInverter
from utils import (compute_metrics, print_metrics, plot_gravity_field,
                   plot_bathymetry, plot_residual, plot_scatter_compare,
                   plot_error_histogram, interpolate_nans, bandpass_filter, compute_vgg)

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output_real"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NC_ENGINE = "h5netcdf"


def load_ssv_region(lon_range, lat_range):
    """加载指定区域的 SSV33.1 重力异常和 VGG。处理 0-360 → -180-180 经度转换。"""
    grav_path = DATA_DIR / "ssv32" / "grav_33.1.nc"
    curv_path = DATA_DIR / "ssv32" / "curv_33.1.nc"

    ds_g = xr.open_dataset(str(grav_path), engine=NC_ENGINE)
    ds_c = xr.open_dataset(str(curv_path), engine=NC_ENGINE)

    lat_slice = slice(lat_range[0], lat_range[1])
    lon0 = (lon_range[0] + 360) % 360
    lon1 = (lon_range[1] + 360) % 360

    if lon0 < lon1:
        gravity = ds_g.z.sel(lon=slice(lon0, lon1), lat=lat_slice).values
        vgg = ds_c.z.sel(lon=slice(lon0, lon1), lat=lat_slice).values
        lon = ds_g.lon.sel(lon=slice(lon0, lon1)).values
    else:
        g1 = ds_g.z.sel(lon=slice(lon0, 360), lat=lat_slice).values
        g2 = ds_g.z.sel(lon=slice(0, lon1), lat=lat_slice).values
        gravity = np.concatenate([g1, g2], axis=1)
        v1 = ds_c.z.sel(lon=slice(lon0, 360), lat=lat_slice).values
        v2 = ds_c.z.sel(lon=slice(0, lon1), lat=lat_slice).values
        vgg = np.concatenate([v1, v2], axis=1)
        l1 = ds_g.lon.sel(lon=slice(lon0, 360)).values
        l2 = ds_g.lon.sel(lon=slice(0, lon1)).values
        lon = np.concatenate([l1, l2])

    lat = ds_g.lat.sel(lat=lat_slice).values
    ds_g.close()
    ds_c.close()
    return gravity, vgg, lon, lat


def load_gebco_region(lon_range, lat_range):
    """加载指定区域的 GEBCO sub-ice 海底地形。"""
    gebco_path = DATA_DIR / "gebco" / "GEBCO_2026_sub_ice" / "GEBCO_2026_sub_ice.nc"
    ds = xr.open_dataset(str(gebco_path), engine=NC_ENGINE)

    var_name = [v for v in ds.variables if len(ds[v].dims) >= 2][0]
    ds_var = ds[var_name]

    pad = 0.5
    bathy = ds_var.sel(lon=slice(lon_range[0] - pad, lon_range[1] + pad),
                        lat=slice(lat_range[0] - pad, lat_range[1] + pad)).values
    lon_b = ds.lon.sel(lon=slice(lon_range[0] - pad, lon_range[1] + pad)).values
    lat_b = ds.lat.sel(lat=slice(lat_range[0] - pad, lat_range[1] + pad)).values
    ds.close()
    return bathy, lon_b, lat_b


def _read_gmt_grd(filepath):
    """读取 .grd/.grd.gz 文件（NetCDF3 或 GMT 二进制），返回数据+经纬度。"""
    spath = str(filepath)
    tmp = tempfile.NamedTemporaryFile(suffix=".grd", delete=False,
                                      prefix=f"grd_{os.getpid()}_{hash(spath) & 0x7FFFFFFF}_")
    tmppath = tmp.name
    try:
        if spath.endswith(".gz"):
            tmp.write(gzip.open(spath, "rb").read())
        else:
            tmp.write(open(spath, "rb").read())
        tmp.close()

        ds = nc.Dataset(tmppath, "r")
        var_names = [v for v in ds.variables if len(ds[v].dimensions) >= 2]
        if var_names:
            vname = var_names[0]
            v = ds.variables[vname]
            data = v[:].copy()
            lon = ds.variables["lon"][:].copy() if "lon" in ds.variables else None
            lat = ds.variables["lat"][:].copy() if "lat" in ds.variables else None
            if lon is None:
                lon = ds.variables["longitude"][:].copy()
            if lat is None:
                lat = ds.variables["latitude"][:].copy()
            ds.close()
            for _ in range(5):
                try:
                    os.unlink(tmppath)
                    break
                except OSError:
                    time.sleep(0.1)
            return np.squeeze(data), np.squeeze(lon), np.squeeze(lat)
        ds.close()

        with open(tmppath, "rb") as f:
            hdr = f.read(892).decode("latin-1", errors="ignore")
            nx = int(re.search(r"n_columns\s*=\s*(\d+)", hdr).group(1))
            ny = int(re.search(r"n_rows\s*=\s*(\d+)", hdr).group(1))
            xmin = float(re.search(r"x_min\s*=\s*([\-\d.]+)", hdr).group(1))
            xmax = float(re.search(r"x_max\s*=\s*([\-\d.]+)", hdr).group(1))
            ymin = float(re.search(r"y_min\s*=\s*([\-\d.]+)", hdr).group(1))
            ymax = float(re.search(r"y_max\s*=\s*([\-\d.]+)", hdr).group(1))
            data = np.frombuffer(f.read(nx * ny * 4), dtype=np.float32).reshape(ny, nx)
        os.unlink(tmppath)
        return data, np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
    finally:
        for _ in range(10):
            try:
                if os.path.exists(tmppath):
                    os.unlink(tmppath)
                break
            except OSError:
                time.sleep(0.05)


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
                "faa", "free_air", "free-air", "_gravity", "residual", "maba", "rmba", "bouguer"])
            is_bathy = any(kw in fname for kw in ["_mb", "bathy", "0.00025"])

            if not (is_gravity or is_bathy):
                continue

            try:
                data, lon, lat = _read_gmt_grd(str(f))
                if data is None:
                    continue

                lon_2d, lat_2d = np.meshgrid(lon, lat)
                mask = ((lon_2d >= lon_grid[0]) & (lon_2d <= lon_grid[-1]) &
                        (lat_2d >= lat_grid[0]) & (lat_2d <= lat_grid[-1]))
                if mask.sum() == 0:
                    continue

                valid = mask & ~np.isnan(data)
                idx = np.where(valid)
                for i, j in zip(idx[0], idx[1]):
                    pt = (lon[j], lat[i], data[i, j])
                    if is_gravity and abs(data[i, j]) < 5000:
                        gravity_points.append(pt)
                    elif is_bathy:
                        bathy_points.append(pt)

                print(f"  Loaded {f.name}: {valid.sum()} points")
            except Exception:
                pass

    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    ship_grav = np.full(lon_mesh.shape, np.nan)
    ship_bathy = np.full(lon_mesh.shape, np.nan)

    if gravity_points:
        pts = np.array(gravity_points)
        ship_grav = griddata(pts[:, :2], pts[:, 2], (lon_mesh, lat_mesh), method="nearest")
    if bathy_points:
        pts = np.array(bathy_points)
        ship_bathy = griddata(pts[:, :2], pts[:, 2], (lon_mesh, lat_mesh), method="nearest")

    return ship_grav, ship_bathy


def run_pipeline(lon_range=(140, 150), lat_range=(30, 40), upscale=4):
    """完整管线：加载数据 → KF重力重建 → KF水深反演 → 评估+图表。"""
    print("=" * 60)
    print("  Kalman Filter Gravity Field Modeling - Real Data")
    print("=" * 60)
    print(f"  Region: lon=[{lon_range[0]}, {lon_range[1]}], lat=[{lat_range[0]}, {lat_range[1]}]")

    print("\n[1/3] Loading SSV33.1 satellite data...")
    gravity_lr, vgg_lr, lon_lr, lat_lr = load_ssv_region(lon_range, lat_range)
    print(f"  Gravity: {gravity_lr.shape}")

    print("\n[2/3] Loading GEBCO bathymetry...")
    bathy_gebco, lon_b, lat_b = load_gebco_region(lon_range, lat_range)
    print(f"  GEBCO: {bathy_gebco.shape}")

    from scipy.ndimage import zoom as sp_zoom
    zoom_fac = (gravity_lr.shape[0] / bathy_gebco.shape[0],
                gravity_lr.shape[1] / bathy_gebco.shape[1])
    bathy_prior_lr = sp_zoom(bathy_gebco, zoom_fac, order=1)

    print("\n[3/3] Loading shipborne data...")
    ny_lr, nx_lr = gravity_lr.shape
    ny_hr, nx_hr = ny_lr * upscale, nx_lr * upscale
    lon_hr = np.linspace(lon_range[0], lon_range[1], nx_hr)
    lat_hr = np.linspace(lat_range[0], lat_range[1], ny_hr)
    ship_grav, ship_bathy = load_shipborne_gravity(lon_hr, lat_hr)
    n_ship = np.sum(~np.isnan(ship_grav)) if ship_grav is not None else 0
    print(f"  Shipborne gravity points: {n_ship}")

    print("\n[4/5] Kalman Filter Gravity Reconstruction...")
    reconstructor = KalmanGravityReconstructor(
        grid_shape=(ny_hr, nx_hr), region_size_deg=lat_range[1] - lat_range[0],
        lr_grid_shape=(ny_lr, nx_lr))
    reconstructor.initialize(
        gravity_lr=gravity_lr, gravity_ship=ship_grav if n_ship > 0 else None,
        process_noise_std=2.0, sat_noise_std=3.0, ship_noise_std=1.0)

    t0 = time.time()
    gravity_hr = reconstructor.run(n_iterations=6, smooth_sigma=0.8)
    print(f"  Done in {time.time()-t0:.0f}s")

    gravity_bilinear = zoom(gravity_lr, (upscale, upscale), order=1)

    print("\n[5/5] Kalman Filter Bathymetry Inversion...")
    inverter = KalmanBathymetryInverter(
        grid_shape=(ny_hr, nx_hr), region_size_deg=lat_range[1] - lat_range[0],
        lat_center=(lat_range[0] + lat_range[1]) / 2.0)

    bathy_prior_hr = sp_zoom(bathy_gebco, (ny_hr / bathy_gebco.shape[0],
                                            nx_hr / bathy_gebco.shape[1]), order=1)
    inverter.depth_min = max(-11000, np.nanmin(bathy_prior_hr) - 500)
    inverter.depth_max = min(1000, np.nanmax(bathy_prior_hr) + 500)
    inverter.apply_flexural_isostasy(Te=10e3)
    inverter.initialize(gravity_reconstructed=gravity_hr,
                        gravity_ship=ship_grav if n_ship > 0 else None,
                        bathy_ship=ship_bathy, bathymetry_prior=bathy_prior_hr,
                        process_noise_std=80.0, observation_noise_std=5.0)

    bathymetry_kf = inverter.run(n_iterations=5, smooth_sigma=1.0)
    bathymetry_trad = inverter.gravity_to_bathymetry_linear(
        gravity_hr, k_min=2 * np.pi / 160000, k_max=2 * np.pi / 20000)

    bathy_truth_hr = sp_zoom(bathy_gebco, (ny_hr / bathy_gebco.shape[0],
                                            nx_hr / bathy_gebco.shape[1]), order=1)
    m_g_kf = compute_metrics(gravity_hr, gravity_bilinear)
    m_b_kf = compute_metrics(bathymetry_kf, bathy_truth_hr)
    m_b_trad = compute_metrics(bathymetry_trad, bathy_truth_hr)
    m_g_ship = compute_metrics(gravity_hr, ship_grav) if n_ship > 0 else None

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Gravity KF mean/std: {gravity_hr.mean():.1f} / {gravity_hr.std():.1f} mGal")
    if m_g_ship:
        print(f"  Gravity KF vs Ship RMSE: {m_g_ship['RMSE']:.1f} mGal")
    print(f"  Bathymetry KF mean: {bathymetry_kf.mean():.0f} m")
    print(f"  Bathymetry RMSE vs GEBCO: {m_b_kf['RMSE']:.0f} m (KF) vs {m_b_trad['RMSE']:.0f} m (Parker)")

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
