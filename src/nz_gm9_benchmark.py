"""新西兰 GM9 海域断层：局部 5–20 km 频带的断裂验证。"""
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
LON, LAT = (170.3, 172.1), (-46.4, -45.0)


def main():
    g_lr, _, _, _ = load_ssv_region(LON, LAT)
    ny, nx = g_lr.shape
    up = 4
    lon = cell_centers(LON[0], LON[1], nx * up)
    lat = cell_centers(LAT[0], LAT[1], ny * up)
    bathy = load_bathy_15s(LON, LAT, lon, lat)
    ssv_hr = sp_zoom(g_lr, (up, up), order=1)
    ship = np.full(ssv_hr.shape, np.nan)
    m0 = np.zeros(ssv_hr.shape, dtype=bool)
    kf = BathyPriorKF(ssv_hr.shape, g_lr.shape, (1.8, 1.4), alpha=0.5)
    g15 = kf.build(g_lr.astype(float), ship, m0, lon, lat, LON, LAT,
                   upscale=up, bathy_15s=bathy, n_iter=6, smooth_sigma=4.0)

    dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = float(lat[1] - lat[0]) * 111320

    d = json.loads((OUT / "nz_gm9_offshore.json").read_text(encoding="utf-8"))
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
    flon, flat = densify(lines, 0.5)
    ins = ((flon >= LON[0]) & (flon <= LON[1]) &
           (flat >= LAT[0]) & (flat <= LAT[1]))
    flon, flat = flon[ins], flat[ins]
    print(f"lines {len(lines)}, points {len(flon):,}")

    b = bandpass(g15, dx, dy, 5e3, 20e3)
    lm = filter_lineaments(lineament_mask(b, dx, dy, pct=84) & (bathy < 0),
                           min_area=40, min_elong=2.0)
    print("lineaments", lm.sum())
    for r in (1, 2, 5, 10, 20):
        st = fault_match(lm, flon, flat, lon, lat, radius_km=r)
        print(f"r={r:2d}: 覆盖 {st['recall']*100:5.1f}% (随机 {st['recall_random']*100:4.1f}%) "
              f"| 命中 {st['precision']*100:5.1f}% (随机 {st['precision_random']*100:4.1f}%)")


if __name__ == "__main__":
    main()
