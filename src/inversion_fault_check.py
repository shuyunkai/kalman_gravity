"""用反演地形做线性体，与直接用重力做对比（GM9 窗口）。"""
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
from scale_decomposition import bandpass, lineament_mask, filter_lineaments
from bathymetry_inversion import bandpass as bp_inv
from kf_inversion import spectral_kf_inversion
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
    kf = BathyPriorKF(ssv_hr.shape, g_lr.shape, (1.4, 1.8), alpha=0.5)
    g = kf.build(g_lr.astype(float), ship, m0, lon, lat, LON, LAT,
                 upscale=up, bathy_15s=bathy, n_iter=6, smooth_sigma=4.0)
    dx = float(lon[1]-lon[0])*111320*np.cos(np.radians(lat.mean()))
    dy = float(lat[1]-lat[0])*111320

    d = json.loads((OUT / "nz_gm9_offshore.json").read_text(encoding="utf-8"))
    lines = []
    for f in d["features"]:
        geo = f.get("geometry") or {}
        if geo.get("type") != "LineString":
            continue
        a = np.asarray(geo["coordinates"], float)
        if ((a[:,0]>=LON[0])&(a[:,0]<=LON[1])&(a[:,1]>=LAT[0])&(a[:,1]<=LAT[1])).any():
            lines.append(a)
    idx = np.arange(len(lines)); rng = np.random.default_rng(7); rng.shuffle(idx)
    test = [lines[i] for i in idx[len(lines)//2:]]
    tx, ty = densify(test, 0.5)

    # 反演：重力带通 20–200 km
    z0 = float(np.mean(-bathy))
    dg = bp_inv(g, dx, dy, 20e3, 200e3)
    hk, _ = spectral_kf_inversion(dg, dx, dy, 2200.0, z0, prior_rms=2500.0,
                                  n_iter=8, lam_min=25e3, relax=0.6)

    # 两条线状体：重力 5–20 km；反演地形 5–20 km
    for tag, field in (("gravity", g), ("inverted bathy", hk)):
        b = bandpass(field, dx, dy, 5e3, 20e3)
        mask = filter_lineaments(lineament_mask(b, dx, dy, pct=70) & (bathy < 0),
                                 min_area=20, min_elong=1.8)
        st = fault_match(mask, tx, ty, lon, lat, radius_km=5)
        print("%s: lineaments=%d, 5km cover %.1f%% (rand %.1f%%)" % (
            tag, mask.sum(), st["recall"]*100, st["recall_random"]*100))


if __name__ == "__main__":
    main()
