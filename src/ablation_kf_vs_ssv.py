"""消融：KF 15″ 场 vs SSV 普通插值场，同样方法做断层与反演。"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom as sp_zoom

sys.path.insert(0, str(Path(__file__).parent))

from run_pipeline import load_ssv_region
from three_method_compare import load_bathy_15s
from utils import cell_centers
from kf_novel import BathyPriorKF
from bathymetry_inversion import bandpass as bp, parker_inversion
from scale_decomposition import bandpass, lineament_mask, filter_lineaments
from fault_validation import densify, fault_match

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real" / "three_parts"


def fault_compare():
    LON, LAT = (170.3, 172.1), (-46.4, -45.0)
    g_lr, _, _, _ = load_ssv_region(LON, LAT)
    ny, nx = g_lr.shape
    up = 4
    lon = cell_centers(LON[0], LON[1], nx * up)
    lat = cell_centers(LAT[0], LAT[1], ny * up)
    bathy = load_bathy_15s(LON, LAT, lon, lat)
    ssv_hr = sp_zoom(g_lr, (up, up), order=1)
    kf = BathyPriorKF(ssv_hr.shape, g_lr.shape, (1.4, 1.8), alpha=0.5)
    g_kf = kf.build(g_lr.astype(float), np.full(ssv_hr.shape, np.nan),
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
    test = [lines[i] for i in idx[len(lines) // 2:]]
    tx, ty = densify(test, 0.5)

    print("fault: field  cover5  cover10  px")
    for name, field in (("KF", g_kf), ("SSV", ssv_hr)):
        b = bandpass(field, dx, dy, 5e3, 20e3)
        m = filter_lineaments(lineament_mask(b, dx, dy, pct=70) & (bathy < 0),
                              20, 1.8)
        c5 = fault_match(m, tx, ty, lon, lat, radius_km=5)["recall"] * 100
        c10 = fault_match(m, tx, ty, lon, lat, radius_km=10)["recall"] * 100
        print(f"{name:6s} {c5:7.1f} {c10:8.1f} {int(m.sum()):5d}")


def inversion_compare():
    f = np.load(ROOT / "output_real" / "test_field_15s_jodc.npz")
    g_kf = f["g15"]; bathy = f["bathy"]; lon = f["lon"]; lat = f["lat"]
    k = np.load(ROOT / "output_real" / "three_parts" / "kf_inputs.npz")
    g_ssv = k["ssv_hr"]
    dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = float(lat[1] - lat[0]) * 111320
    z0 = float(-np.median(bathy))
    target = bp(z0 + bathy, dx, dy, 20e3, 200e3)
    inner = np.zeros_like(bathy, dtype=bool)
    inner[40:-40, 40:-40] = True
    deep = (bathy < -2000) & inner & np.isfinite(target)

    print("inversion: field  20-50  50-100 100-200 20-200 corr")
    for name, g in (("KF", g_kf), ("SSV", g_ssv)):
        dg = bp(g, dx, dy, 20e3, 200e3)
        out = parker_inversion(dg, dx, dy, 2200.0, z0, n_iter=8,
                               lam_min=25e3, relax=0.6, verbose=False)
        h = out[0] if isinstance(out, tuple) else out
        vals = []
        for lo, hi in ((20e3, 50e3), (50e3, 100e3), (100e3, 200e3)):
            ha = bp(h, dx, dy, lo, hi); ta = bp(target, dx, dy, lo, hi)
            mm = deep & np.isfinite(ha) & np.isfinite(ta)
            vals.append(float(np.sqrt(np.mean((ha[mm] - ta[mm]) ** 2))))
        mm = deep & np.isfinite(h)
        full = float(np.sqrt(np.mean((h[mm] - target[mm]) ** 2)))
        corr = float(np.corrcoef(h[mm], target[mm])[0, 1])
        print(f"{name:9s} {vals[0]:6.1f} {vals[1]:7.1f} {vals[2]:7.1f} "
              f"{full:7.1f} {corr:5.2f}")


if __name__ == "__main__":
    fault_compare()
    inversion_compare()
