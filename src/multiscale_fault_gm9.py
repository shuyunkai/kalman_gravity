"""GM9 窗口：多尺度 KF 分量做断裂提取，和带通总场对比。"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom as sp_zoom

sys.path.insert(0, str(Path(__file__).parent))

from run_pipeline import load_ssv_region
from three_method_compare import load_bathy_15s
from utils import cell_centers
from multiscale_kf import MultiScaleKF
from scale_decomposition import bandpass, lineament_mask, filter_lineaments
from fault_validation import densify, fault_match

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real" / "three_parts"
LON, LAT = (170.3, 172.1), (-46.4, -45.0)


def main():
    g_lr, _, _, _ = load_ssv_region(LON, LAT)
    ny, nx = g_lr.shape
    up = 4
    lon = cell_centers(LON[0], LON[1], nx * up)
    lat = cell_centers(LAT[0], LAT[1], ny * up)
    bathy = load_bathy_15s(LON, LAT, lon, lat)
    ssv_hr = sp_zoom(g_lr, (up, up), order=1)
    dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = float(lat[1] - lat[0]) * 111320

    c = np.where(bathy < 0, 0.0688, 0.1120)
    boug = c * bathy
    hp = boug - gaussian_filter(boug, 12.0)

    kf = MultiScaleKF(ssv_hr, hp, sigmas=(16.0, 4.0, 1.0), alpha=0.5)
    ship = np.zeros_like(ssv_hr)
    mask0 = np.zeros_like(ssv_hr, dtype=bool)
    x1, x2, x3 = kf.run(g_lr.astype(float), ship, mask0, n_iter=6)
    total = x1 + x2 + x3
    print("grid", lon[0], lon[-1], lat[0], lat[-1],
          "bathy", np.isfinite(bathy).sum())
    print("components std", float(np.std(x1)), float(np.std(x2)),
          float(np.std(x3)))

    d = json.loads((OUT / "nz_gm9_offshore.json").read_text(encoding="utf-8"))
    lines = []
    for f in d["features"]:
        geo = f.get("geometry") or {}
        if geo.get("type") != "LineString":
            continue
        a = np.asarray(geo["coordinates"], float)
        if ((a[:, 0] >= lon.min()) & (a[:, 0] <= lon.max()) &
                (a[:, 1] >= lat.min()) & (a[:, 1] <= lat.max())).any():
            lines.append(a)
    idx = np.arange(len(lines)); np.random.default_rng(7).shuffle(idx)
    test = [lines[i] for i in idx[len(lines) // 2:]]
    tx, ty = densify(test, 0.5)
    print("lines", len(lines), "test", len(test), "pts", len(tx))

    for tag, field in (("bandpass(total)", bandpass(total, dx, dy, 5e3, 20e3)),
                       ("bandpass(x2+x3)", bandpass(x2 + x3, dx, dy, 5e3, 20e3))):
        m = filter_lineaments(lineament_mask(field, dx, dy, pct=70) & (bathy < 0),
                              min_area=20, min_elong=1.8)
        st = fault_match(m, tx, ty, lon, lat, radius_km=5)
        print(f"{tag}: lineaments={m.sum():,} 5km cover={st['recall']*100:.1f}%")


if __name__ == "__main__":
    main()
