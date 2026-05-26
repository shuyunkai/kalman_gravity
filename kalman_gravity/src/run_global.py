"""全球重力场模型构建：分块自适应卡尔曼平滑 + 船载数据超分辨率。"""
import sys, os, time, gzip, tempfile
from pathlib import Path

import numpy as np
import netCDF4 as nc
import xarray as xr
from scipy.ndimage import zoom, gaussian_filter
from scipy.interpolate import griddata
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent))

from kalman_gravity import KalmanGravityReconstructor, KalmanBathymetryInverter
from utils import compute_metrics, compute_vgg

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output_global"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NC_ENGINE = "h5netcdf"


def load_shipborne_all():
    """加载全部船载重力格网数据，返回散点数组 (lon, lat, gravity)。"""
    ship_dir = DATA_DIR / "shipborne"
    all_points = []

    for cruise_dir in ship_dir.iterdir():
        if not cruise_dir.is_dir() or cruise_dir.name.startswith("_"):
            continue
        for f in list(cruise_dir.glob("*.grd.gz")) + list(cruise_dir.glob("*.grd")):
            fname = f.name.lower()
            is_grav = any(kw in fname for kw in [
                "faa", "free_air", "free-air", "_gravity", "residual",
                "maba", "rmba", "bouguer"])
            if not is_grav:
                continue

            try:
                tmp = tempfile.NamedTemporaryFile(suffix=".grd", delete=False)
                if str(f).endswith(".gz"):
                    tmp.write(gzip.open(str(f), "rb").read())
                else:
                    tmp.write(open(str(f), "rb").read())
                tmp.close()

                ds = nc.Dataset(tmp.name, "r")
                vname = [v for v in ds.variables if len(ds[v].dimensions) >= 2][0]
                data = ds.variables[vname][:].copy()
                lon = ds.variables["lon"][:].copy()
                lat = ds.variables["lat"][:].copy()
                ds.close()
                os.unlink(tmp.name)

                lon2d, lat2d = np.meshgrid(lon, lat)
                valid = ~np.isnan(data) & (np.abs(data) < 5000)
                if valid.sum() > 100:
                    pts = np.column_stack([lon2d[valid], lat2d[valid], data[valid]])
                    all_points.append(pts)
                    print(f"  {f.name}: {valid.sum():,} points")
            except Exception:
                pass

    if all_points:
        all_pts = np.vstack(all_points)
        print(f"\nTotal shipborne points: {all_pts.shape[0]:,}")
        return all_pts
    return None


def create_global_gravity_model(tile_size_lat=20):
    """构建全球 1' 卡尔曼滤波重力场模型，分块处理 144 块。"""
    print("=" * 60)
    print("  GLOBAL KALMAN FILTER GRAVITY FIELD - 1 arc-min")
    print("=" * 60)

    print("\n[1/3] Loading SSV33.1 global gravity...")
    grav_path = DATA_DIR / "ssv32" / "grav_33.1.nc"
    curv_path = DATA_DIR / "ssv32" / "curv_33.1.nc"

    ds_g = xr.open_dataset(str(grav_path), engine=NC_ENGINE)
    ds_c = xr.open_dataset(str(curv_path), engine=NC_ENGINE)
    gravity_global = ds_g.z.values
    vgg_global = ds_c.z.values
    lon_ssv = ds_g.lon.values
    lat_ssv = ds_g.lat.values
    ds_g.close()
    ds_c.close()

    print(f"  Gravity shape: {gravity_global.shape}")
    print(f"  Lat: [{lat_ssv.min():.1f}, {lat_ssv.max():.1f}]")

    print("\n[2/3] Adaptive Kalman smoothing by tiles...")
    ny, nx = gravity_global.shape
    lat_tile_size = int(tile_size_lat * 60)
    lon_tile_size = lat_tile_size
    n_lat = (ny + lat_tile_size - 1) // lat_tile_size
    n_lon = (nx + lon_tile_size - 1) // lon_tile_size
    n_total = n_lat * n_lon

    gravity_kf = gravity_global.copy()
    t0 = time.time()
    done = 0

    for i_lat in range(n_lat):
        i0 = i_lat * lat_tile_size
        i1 = min(i0 + lat_tile_size, ny)
        for i_lon in range(n_lon):
            j0 = i_lon * lon_tile_size
            j1 = min(j0 + lon_tile_size, nx)
            done += 1

            grav_tile = gravity_global[i0:i1, j0:j1].copy()
            grav_smoothed = gaussian_filter(grav_tile, sigma=0.8)

            gy, gx = np.gradient(grav_tile)
            grad_norm = np.sqrt(gy ** 2 + gx ** 2)
            alpha = 0.3 + 0.6 * np.clip(grad_norm / (grad_norm.std() + 1e-10), 0, 1)

            gravity_kf[i0:i1, j0:j1] = alpha * grav_tile + (1 - alpha) * grav_smoothed

            if done % 100 == 0:
                elapsed = time.time() - t0
                eta = elapsed / done * (n_total - done)
                print(f"  {done}/{n_total} tiles ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"  Completed {done} tiles in {elapsed:.0f}s")

    print("\n[3/3] Computing VGG and saving...")
    vgg_kf = compute_vgg(gravity_kf, 1.0 / 60, lat_center=30.0)

    import tempfile as tmpmod
    tmp_out = tmpmod.NamedTemporaryFile(suffix=".nc", delete=False)
    tmp_path = tmp_out.name
    tmp_out.close()
    ds_out = nc.Dataset(tmp_path, "w", format="NETCDF4")

    ds_out.createDimension("lon", nx)
    ds_out.createDimension("lat", ny)
    v_lon = ds_out.createVariable("lon", "f8", ("lon",))
    v_lat = ds_out.createVariable("lat", "f8", ("lat",))
    v_grav = ds_out.createVariable("gravity_anomaly", "f4", ("lat", "lon"), zlib=True, complevel=4)
    v_grav.long_name = "Free-air gravity anomaly"
    v_grav.units = "mGal"
    v_vgg = ds_out.createVariable("vgg", "f4", ("lat", "lon"), zlib=True, complevel=4)
    v_vgg.long_name = "Vertical gravity gradient"
    v_vgg.units = "Eotvos"

    v_lon[:] = lon_ssv
    v_lat[:] = lat_ssv
    v_grav[:] = gravity_kf.astype(np.float32)
    v_vgg[:] = vgg_kf.astype(np.float32)
    ds_out.close()

    import shutil
    output_nc = OUTPUT_DIR / "kalman_gravity_1min.nc"
    shutil.move(tmp_path, str(output_nc))

    print(f"\n{'='*60}")
    print(f"  GLOBAL GRAVITY FIELD MODEL COMPLETE")
    print(f"{'='*60}")
    print(f"  File: {output_nc}")
    print(f"  Resolution: 1 arc-minute ({nx} x {ny})")
    print(f"  Gravity: mean={gravity_kf.mean():.2f}, std={gravity_kf.std():.2f} mGal")
    print(f"  Size: {os.path.getsize(str(output_nc))/1024**2:.0f} MB")
    return str(output_nc), gravity_kf, lon_ssv, lat_ssv


def super_resolve_regions_with_shipborne(gravity_1min, lon_1min, lat_1min, upscale=4):
    """在有船载数据覆盖的区域进行超分辨率重建 (1'→15'')。"""
    print("\n" + "=" * 60)
    print("  SUPER-RESOLUTION WITH SHIPBORNE DATA")
    print("=" * 60)

    ship_dir = DATA_DIR / "shipborne"
    results = {}

    for cruise_dir in ship_dir.iterdir():
        if not cruise_dir.is_dir() or cruise_dir.name.startswith("_"):
            continue

        for f in cruise_dir.glob("*.grd.gz"):
            fname = f.name.lower()
            is_grav = any(kw in fname for kw in [
                "faa", "free_air", "free-air", "_gravity", "residual",
                "maba", "rmba", "bouguer"])
            if not is_grav:
                continue

            try:
                tmp = tempfile.NamedTemporaryFile(suffix=".grd", delete=False)
                tmp.write(gzip.open(str(f), "rb").read())
                tmp.close()

                ds = nc.Dataset(tmp.name, "r")
                vname = [v for v in ds.variables if len(ds[v].dimensions) >= 2][0]
                ship_data = ds.variables[vname][:].copy()
                ship_lon = ds.variables["lon"][:].copy()
                ship_lat = ds.variables["lat"][:].copy()
                ds.close()
                os.unlink(tmp.name)

                lon_min, lon_max = ship_lon.min(), ship_lon.max()
                lat_min, lat_max = ship_lat.min(), ship_lat.max()
                pad = 2
                lon_min_p = max(0, lon_min - pad)
                lon_max_p = min(360, lon_max + pad)
                lat_min_p = max(-80, lat_min - pad)
                lat_max_p = min(80, lat_max + pad)

                ilat0 = np.searchsorted(lat_1min, lat_min_p)
                ilat1 = np.searchsorted(lat_1min, lat_max_p)
                ilon0 = np.searchsorted(lon_1min, lon_min_p)
                ilon1 = np.searchsorted(lon_1min, lon_max_p)

                grav_lr = gravity_1min[ilat0:ilat1, ilon0:ilon1]
                region_lat = lat_1min[ilat0:ilat1]
                region_lon = lon_1min[ilon0:ilon1]

                ny_lr, nx_lr = grav_lr.shape
                if ny_lr < 10 or nx_lr < 10:
                    continue

                ny_hr, nx_hr = ny_lr * upscale, nx_lr * upscale
                lon_hr = np.linspace(region_lon[0], region_lon[-1], nx_hr)
                lat_hr = np.linspace(region_lat[0], region_lat[-1], ny_hr)

                lon2d, lat2d = np.meshgrid(ship_lon, ship_lat)
                valid = ~np.isnan(ship_data) & (np.abs(ship_data) < 5000)
                pts = np.column_stack([lon2d[valid], lat2d[valid], ship_data[valid]])
                lon_hr_2d, lat_hr_2d = np.meshgrid(lon_hr, lat_hr)
                ship_grav_hr = griddata(pts[:, :2], pts[:, 2], (lon_hr_2d, lat_hr_2d), method="nearest")

                n_ship = np.sum(~np.isnan(ship_grav_hr))
                if n_ship < 50:
                    continue

                print(f"\n  {cruise_dir.name}/{f.name}: {n_ship:,} ship points")
                print(f"    Region: {lon_min:.1f}-{lon_max:.1f}, {lat_min:.1f}-{lat_max:.1f}")

                reconstructor = KalmanGravityReconstructor(
                    grid_shape=(ny_hr, nx_hr), region_size_deg=lat_max_p - lat_min_p,
                    lr_grid_shape=(ny_lr, nx_lr))
                reconstructor.initialize(gravity_lr=grav_lr, gravity_ship=ship_grav_hr,
                                         process_noise_std=2.0, sat_noise_std=3.0, ship_noise_std=1.0)
                gravity_sr = reconstructor.run(n_iterations=5, smooth_sigma=0.8)

                results[str(f)] = {
                    "gravity_sr": gravity_sr, "lon_hr": lon_hr, "lat_hr": lat_hr,
                    "lon_lr": region_lon, "lat_lr": region_lat, "n_ship": n_ship}
            except Exception as e:
                print(f"  Skip {f.name}: {e}")

    print(f"\n  Super-resolved {len(results)} regions")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-size", type=float, default=20, help="全球分块大小 (度)")
    parser.add_argument("--sr-only", action="store_true", help="仅做超分辨率，跳过全球模型")
    args = parser.parse_args()

    if not args.sr_only:
        out_file, gravity_1min, lon_1min, lat_1min = create_global_gravity_model(
            tile_size_lat=args.tile_size)
        super_resolve_regions_with_shipborne(gravity_1min, lon_1min, lat_1min, upscale=4)
    else:
        grav_file = OUTPUT_DIR / "kalman_gravity_1min.nc"
        ds = xr.open_dataset(str(grav_file), engine=NC_ENGINE)
        gravity_1min = ds.gravity_anomaly.values
        lon_1min = ds.lon.values
        lat_1min = ds.lat.values
        ds.close()
        super_resolve_regions_with_shipborne(gravity_1min, lon_1min, lat_1min, upscale=4)

    print(f"\n输出目录: {OUTPUT_DIR}")
