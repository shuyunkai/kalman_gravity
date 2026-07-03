"""区域数据加载器: SSV33.1 卫星重力 + GEBCO 海底地形 + 船载重力.

仅保留加载函数, KF 重建逻辑在 kf_novel.py (BathyPriorKF).
"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))

from utils import read_gmt_grd

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
    """加载指定区域的 GEBCO sub_ice 海底地形。"""
    gebco_path = DATA_DIR / "gebco" / "GEBCO_2026_sub_ice" / "GEBCO_2026_sub_ice.nc"
    ds = xr.open_dataset(str(gebco_path), engine=NC_ENGINE)

    var_name = [v for v in ds.variables if len(ds[v].dims) >= 2][0]
    ds_var = ds[var_name]

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
    """加载船载重力格网 (旧 data/shipborne/ 路径), 精确像素分配到目标网格。

    注: 主流程已改用 jp_data/ 原始航次 (见 independent_validation_v2.load_all_cruises_split).
    """
    ship_dir = DATA_DIR / "shipborne"
    gravity_points = []
    bathy_points = []

    for cruise_dir in ship_dir.iterdir():
        if not cruise_dir.is_dir() or cruise_dir.name.startswith("_"):
            continue
        for f in list(cruise_dir.glob("*.grd.gz")) + list(cruise_dir.glob("*.grd")):
            fname = f.name.lower()
            is_gravity = any(kw in fname for kw in [
                "faa", "free_air", "free-air", "_gravity", "residual",
                "maba", "rmba", "bouguer", "ba."])
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
        j_idx = np.round((pts[:, 0] - lon_grid[0]) / (lon_grid[1] - lon_grid[0])).astype(int)
        i_idx = np.round((pts[:, 1] - lat_grid[0]) / (lat_grid[1] - lat_grid[0])).astype(int)
        valid = (j_idx >= 0) & (j_idx < ship_grav.shape[1]) & \
                (i_idx >= 0) & (i_idx < ship_grav.shape[0])
        ship_grav[i_idx[valid], j_idx[valid]] = pts[valid, 2]
        print(f"  Gravity: {valid.sum():,} exact pixel assignments")

    if bathy_points:
        pts = np.vstack(bathy_points)
        j_idx = np.round((pts[:, 0] - lon_grid[0]) / (lon_grid[1] - lon_grid[0])).astype(int)
        i_idx = np.round((pts[:, 1] - lat_grid[0]) / (lat_grid[1] - lat_grid[0])).astype(int)
        valid = (j_idx >= 0) & (j_idx < ship_bathy.shape[1]) & \
                (i_idx >= 0) & (i_idx < ship_bathy.shape[0])
        ship_bathy[i_idx[valid], j_idx[valid]] = pts[valid, 2]
        print(f"  Bathymetry: {valid.sum():,} exact pixel assignments")

    return ship_grav, ship_bathy
