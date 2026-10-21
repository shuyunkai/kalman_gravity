"""爱琴海：GEM 断层（275 条）+ KF + 5–20 km 带的断裂验证。"""
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
LON, LAT = (23.0, 26.0), (38.0, 40.5)


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
    kf = BathyPriorKF(ssv_hr.shape, g_lr.shape, (5.0, 4.0), alpha=0.5)
    g = kf.build(g_lr.astype(float), ship, m0, lon, lat, LON, LAT,
                 upscale=up, bathy_15s=bathy, n_iter=6, smooth_sigma=4.0)
    dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = float(lat[1] - lat[0]) * 111320

    d = json.loads((OUT / "gem_active_faults.geojson").read_text(encoding="utf-8"))
    lines = []
    for f in d["features"]:
        geo = f.get("geometry") or {}
        if geo.get("type") != "LineString":
            continue
        a = np.asarray(geo["coordinates"], float)
        ins = ((a[:, 0] >= LON[0]) & (a[:, 0] <= LON[1]) &
               (a[:, 1] >= LAT[0]) & (a[:, 1] <= LAT[1]))
        if ins.any():
            lines.append(a)
    print("GEM lines in bbox", len(lines))

    fb = RegularGridInterpolator((lat, lon), bathy, bounds_error=False, fill_value=1.0)
    sea = [a for a in lines if fb([[a.mean(axis=0)[1], a.mean(axis=0)[0]]])[0] < 0]
    print("offshore lines", len(sea))
    tx, ty = densify(sea, 0.5)
    print("offshore points", len(tx))

    b = bandpass(g, dx, dy, 5e3, 20e3)
    mask = filter_lineaments(lineament_mask(b, dx, dy, pct=70) & (bathy < 0),
                             min_area=20, min_elong=1.8)
    print("lineaments", mask.sum())
    for r in (2, 5, 10, 20):
        st = fault_match(mask, tx, ty, lon, lat, radius_km=r)
        print("r=%d cover %.1f (rand %.1f) prec %.1f (rand %.1f)" % (
            r, st["recall"]*100, st["recall_random"]*100,
            st["precision"]*100, st["precision_random"]*100))


if __name__ == "__main__":
    main()

