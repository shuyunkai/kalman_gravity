"""同一份 SSV 分别以 .nc / .grd 输入，跑 GM9 断层与日本反演对比。"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import zoom as sp_zoom

sys.path.insert(0, str(Path(__file__).parent))

from run_pipeline import load_ssv_region, DATA_DIR
from three_method_compare import load_bathy_15s, load_region_data
from utils import cell_centers
from kf_novel import BathyPriorKF
from scale_decomposition import bandpass, lineament_mask, filter_lineaments
from fault_validation import densify, fault_match
from bathymetry_inversion import bandpass as bp, parker_inversion
import json

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real" / "three_parts"


def read_grd(path, lon_range, lat_range):
    ds = xr.open_dataset(path)
    lon = ds["lon"].values
    lat = ds["lat"].values
    z = ds["z"].values
    ds.close()
    if lon.max() > 180:
        lon = (lon + 180) % 360 - 180
        order = np.argsort(lon)
        lon = lon[order]
        z = z[:, order]
    i0 = np.searchsorted(lon, lon_range[0])
    i1 = np.searchsorted(lon, lon_range[1])
    j0 = np.searchsorted(lat, lat_range[0])
    j1 = np.searchsorted(lat, lat_range[1])
    return z[j0:j1, i0:i1]


def main():
    # 1) 导出 .grd（GMT 可读的 netCDF 网格）
    for name, rng in (("gm9", ((170.3, 172.1), (-46.4, -45.0))),
                      ("japan", ((141.0, 149.0), (32.0, 40.0)))):
        ds = xr.open_dataset(str(DATA_DIR / "ssv32" / "grav_33.1.nc"))
        lon = ds["lon"].values
        lat = ds["lat"].values
        z = ds["z"].values
        if lon.max() > 180:
            lon = (lon + 180) % 360 - 180
            order = np.argsort(lon)
            lon = lon[order]
            z = z[:, order]
        i0 = np.searchsorted(lon, rng[0][0]); i1 = np.searchsorted(lon, rng[0][1])
        j0 = np.searchsorted(lat, rng[1][0]); j1 = np.searchsorted(lat, rng[1][1])
        sub = xr.Dataset(
            {"z": (("lat", "lon"), z[j0:j1, i0:i1])},
            coords={"lat": lat[j0:j1], "lon": lon[i0:i1]},
        )
        sub["z"].attrs["units"] = "mGal"
        p = DATA_DIR / "ssv32" / f"grav_33.1_{name}.grd"
        sub.to_netcdf(p)
        sub.close(); ds.close()
        print("saved", p)

    # 2) GM9 断层
    LON, LAT = (170.3, 172.1), (-46.4, -45.0)
    g_nc, _, _, _ = load_ssv_region(LON, LAT)
    g_grd = read_grd(DATA_DIR / "ssv32" / "grav_33.1_gm9.grd", LON, LAT)
    print("gm9 max|diff|", float(np.max(np.abs(g_nc - g_grd))))

    ny, nx = g_nc.shape
    up = 4
    lon = cell_centers(LON[0], LON[1], nx * up)
    lat = cell_centers(LAT[0], LAT[1], ny * up)
    bathy = load_bathy_15s(LON, LAT, lon, lat)
    dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = float(lat[1] - lat[0]) * 111320

    lines = []
    for f in json.loads((OUT / "nz_gm9_offshore.json").read_text(encoding="utf-8"))["features"]:
        geo = f.get("geometry") or {}
        if geo.get("type") != "LineString":
            continue
        a = np.asarray(geo["coordinates"], float)
        if ((a[:, 0] >= LON[0]) & (a[:, 0] <= LON[1]) &
                (a[:, 1] >= LAT[0]) & (a[:, 1] <= LAT[1])).any():
            lines.append(a)
    idx = np.arange(len(lines)); np.random.default_rng(7).shuffle(idx)
    tx, ty = densify([lines[i] for i in idx[len(lines) // 2:]], 0.5)

    for tag, g_lr in (("nc", g_nc), ("grd", g_grd)):
        ssv_hr = sp_zoom(g_lr, (up, up), order=1)
        kf = BathyPriorKF(ssv_hr.shape, g_lr.shape, (1.4, 1.8), alpha=0.5)
        g = kf.build(g_lr.astype(float), np.full(ssv_hr.shape, np.nan),
                     np.zeros(ssv_hr.shape, dtype=bool), lon, lat, LON, LAT,
                     upscale=up, bathy_15s=bathy, n_iter=6, smooth_sigma=4.0)
        b = bandpass(g, dx, dy, 5e3, 20e3)
        m = filter_lineaments(lineament_mask(b, dx, dy, pct=70) & (bathy < 0),
                              20, 1.8)
        st = fault_match(m, tx, ty, lon, lat, radius_km=5)
        print(f"fault {tag}: 5km {st['recall']*100:.1f}%")

    # 3) 日本反演
    LON2, LAT2 = (141.0, 149.0), (32.0, 40.0)
    g_nc2, _, _, _ = load_ssv_region(LON2, LAT2)
    g_grd2 = read_grd(DATA_DIR / "ssv32" / "grav_33.1_japan.grd", LON2, LAT2)
    print("japan max|diff|", float(np.max(np.abs(g_nc2 - g_grd2))))

    data = load_region_data(LON2, LAT2, val_cruise="MR22-04_gravity_dmo", upscale=4)
    data["gravity_lr"] = g_grd2
    kf = BathyPriorKF(data["ssv_hr"].shape, data["gravity_lr"].shape,
                      (8.0, 8.0), ship_noise_std=10.0, alpha=0.5)
    gk = kf.build(data["gravity_lr"].astype(float), data["ship_grav"],
                  data["train_mask"], data["lon_hr"], data["lat_hr"],
                  LON2, LAT2, upscale=4, bathy_15s=data["bathy_15s"],
                  n_iter=6, smooth_sigma=4.0)
    dx2 = float(data["lon_hr"][1] - data["lon_hr"][0]) * 111320 * np.cos(
        np.radians(data["lat_hr"].mean()))
    dy2 = float(data["lat_hr"][1] - data["lat_hr"][0]) * 111320
    z0 = float(-np.median(data["bathy_15s"]))
    t = bp(z0 + data["bathy_15s"], dx2, dy2, 20e3, 200e3)
    inner = np.zeros_like(t, dtype=bool); inner[40:-40, 40:-40] = True
    deep = (data["bathy_15s"] < -2000) & inner & np.isfinite(t)
    out = parker_inversion(bp(gk, dx2, dy2, 20e3, 200e3), dx2, dy2, 2200.0,
                           z0, n_iter=8, lam_min=25e3, relax=0.6, verbose=False)
    h = out[0] if isinstance(out, tuple) else out
    vals = []
    for lo, hi in ((20e3, 50e3), (50e3, 100e3)):
        ha = bp(h, dx2, dy2, lo, hi); ta = bp(t, dx2, dy2, lo, hi)
        mm = deep & np.isfinite(ha) & np.isfinite(ta)
        vals.append(float(np.sqrt(np.mean((ha[mm] - ta[mm]) ** 2))))
    print(f"inversion grd: 20-50 {vals[0]:.1f} m, 50-100 {vals[1]:.1f} m")


if __name__ == "__main__":
    main()
