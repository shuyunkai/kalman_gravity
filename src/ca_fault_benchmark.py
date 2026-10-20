"""加州海域：CGS 断层活动图 vs 重力线性体（同一套 KF + 带通流程）。"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom as sp_zoom
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).parent))

from run_pipeline import load_ssv_region
from three_method_compare import load_bathy_15s
from utils import cell_centers
from kf_novel import BathyPriorKF
from scale_decomposition import bandpass, lineament_mask, filter_lineaments
from fault_validation import densify, fault_match

OUT = Path(__file__).parent.parent / "output_real" / "three_parts"
LON, LAT = (-125.0, -121.0), (32.0, 36.0)


def main():
    g_lr, _, lon_lr, lat_lr = load_ssv_region(LON, LAT)
    ny, nx = g_lr.shape
    up = 4
    lon = cell_centers(LON[0], LON[1], nx * up)
    lat = cell_centers(LAT[0], LAT[1], ny * up)
    bathy = load_bathy_15s(LON, LAT, lon, lat)
    print("SSV", g_lr.shape, "bathy", bathy.min(), bathy.max())

    ssv_hr = sp_zoom(g_lr, (up, up), order=1)
    ship = np.full(ssv_hr.shape, np.nan)
    mask0 = np.zeros(ssv_hr.shape, dtype=bool)
    kf = BathyPriorKF(ssv_hr.shape, g_lr.shape, (4.0, 4.0), alpha=0.5)
    g15 = kf.build(g_lr.astype(float), ship, mask0, lon, lat, LON, LAT,
                   upscale=up, bathy_15s=bathy, n_iter=6, smooth_sigma=4.0)

    dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = float(lat[1] - lat[0]) * 111320

    d = json.loads((OUT / "ca_faults.json").read_text(encoding="utf-8"))
    lines = []
    for f in d["features"]:
        g = f.get("geometry") or {}
        if g.get("type") != "LineString":
            continue
        a = np.asarray(g["coordinates"], float)
        ins = ((a[:, 0] >= LON[0]) & (a[:, 0] <= LON[1]) &
               (a[:, 1] >= LAT[0]) & (a[:, 1] <= LAT[1]))
        if ins.any():
            lines.append(a)
    print("CGS faults in bbox", len(lines), "of", len(d["features"]))

    flon, flat = densify(lines, 1.0)
    inside = ((flon >= LON[0]) & (flon <= LON[1]) &
              (flat >= LAT[0]) & (flat <= LAT[1]))
    flon, flat = flon[inside], flat[inside]
    fb = RegularGridInterpolator((lat, lon), bathy, bounds_error=False,
                                 fill_value=1.0)
    bz = fb(np.column_stack([flat, flon]))
    off = bz < 0
    print(f"fault points {len(flon):,}, offshore {off.sum():,}")

    b = bandpass(g15, dx, dy, 10e3, 50e3)
    lm = filter_lineaments(lineament_mask(b, dx, dy, pct=84) & (bathy < 0),
                           min_area=60, min_elong=2.2)
    print("lineaments", lm.sum())

    for tag, (x, y) in (("all", (flon, flat)),
                        ("offshore", (flon[off], flat[off]))):
        print(tag)
        for r in (2, 5, 10, 20, 50):
            st = fault_match(lm, x, y, lon, lat, radius_km=r)
            print(f"  r={r:2d}: 覆盖 {st['recall']*100:5.1f}% "
                  f"(随机 {st['recall_random']*100:4.1f}%) | "
                  f"命中 {st['precision']*100:5.1f}% "
                  f"(随机 {st['precision_random']*100:4.1f}%)")


if __name__ == "__main__":
    main()
