"""α=0（不用 GEBCO 地形先验）对照：断层覆盖与反演精度。"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom as sp_zoom

sys.path.insert(0, str(Path(__file__).parent))

from run_pipeline import load_ssv_region
from three_method_compare import load_bathy_15s, load_region_data
from utils import cell_centers
from kf_novel import BathyPriorKF
from scale_decomposition import bandpass, lineament_mask, filter_lineaments
from fault_validation import densify, fault_match
from bathymetry_inversion import bandpass as bp, parker_inversion

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real" / "three_parts"


def gm9(alpha):
    LON, LAT = (170.3, 172.1), (-46.4, -45.0)
    g_lr, _, _, _ = load_ssv_region(LON, LAT)
    ny, nx = g_lr.shape
    up = 4
    lon = cell_centers(LON[0], LON[1], nx * up)
    lat = cell_centers(LAT[0], LAT[1], ny * up)
    bathy = load_bathy_15s(LON, LAT, lon, lat)
    ssv_hr = sp_zoom(g_lr, (up, up), order=1)
    kf = BathyPriorKF(ssv_hr.shape, g_lr.shape, (1.4, 1.8), alpha=alpha)
    g = kf.build(g_lr.astype(float), np.full(ssv_hr.shape, np.nan),
                 np.zeros(ssv_hr.shape, dtype=bool), lon, lat, LON, LAT,
                 upscale=up, bathy_15s=bathy, n_iter=6, smooth_sigma=4.0)
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
    b = bandpass(g, dx, dy, 5e3, 20e3)
    m = filter_lineaments(lineament_mask(b, dx, dy, pct=70) & (bathy < 0), 20, 1.8)
    st = fault_match(m, tx, ty, lon, lat, radius_km=5)
    print(f"GM9 alpha={alpha}: 5km {st['recall']*100:.1f}%  px {int(m.sum())}")


def japan(alpha):
    LON, LAT = (141.0, 149.0), (32.0, 40.0)
    data = load_region_data(LON, LAT, val_cruise="MR22-04_gravity_dmo", upscale=4)
    kf = BathyPriorKF(data["ssv_hr"].shape, data["gravity_lr"].shape,
                      (8.0, 8.0), ship_noise_std=10.0, alpha=alpha)
    g = kf.build(data["gravity_lr"].astype(float), data["ship_grav"],
                 data["train_mask"], data["lon_hr"], data["lat_hr"], LON, LAT,
                 upscale=4, bathy_15s=data["bathy_15s"], n_iter=6, smooth_sigma=4.0)
    dx = float(data["lon_hr"][1] - data["lon_hr"][0]) * 111320 * np.cos(
        np.radians(data["lat_hr"].mean()))
    dy = float(data["lat_hr"][1] - data["lat_hr"][0]) * 111320
    z0 = float(-np.median(data["bathy_15s"]))
    t = bp(z0 + data["bathy_15s"], dx, dy, 20e3, 200e3)
    inner = np.zeros_like(t, dtype=bool); inner[40:-40, 40:-40] = True
    deep = (data["bathy_15s"] < -2000) & inner & np.isfinite(t)
    out = parker_inversion(bp(g, dx, dy, 20e3, 200e3), dx, dy, 2200.0, z0,
                           n_iter=8, lam_min=25e3, relax=0.6, verbose=False)
    h = out[0] if isinstance(out, tuple) else out
    vals = []
    for lo, hi in ((20e3, 50e3), (50e3, 100e3)):
        ha = bp(h, dx, dy, lo, hi); ta = bp(t, dx, dy, lo, hi)
        mm = deep & np.isfinite(ha) & np.isfinite(ta)
        vals.append(float(np.sqrt(np.mean((ha[mm] - ta[mm]) ** 2))))
    print(f"Japan alpha={alpha}: 20-50 {vals[0]:.1f} m, 50-100 {vals[1]:.1f} m")


if __name__ == "__main__":
    gm9(0.0)
    gm9(0.5)
    japan(0.0)
    japan(0.5)
