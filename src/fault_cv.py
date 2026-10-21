"""GM9：5 折交叉验证不同线状体掩膜的断层覆盖率。"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom as sp_zoom

sys.path.insert(0, str(Path(__file__).parent))

from run_pipeline import load_ssv_region
from three_method_compare import load_bathy_15s
from utils import cell_centers
from kf_novel import BathyPriorKF
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
    hp = c * bathy - gaussian_filter(c * bathy, 12.0)

    kf = BathyPriorKF(ssv_hr.shape, g_lr.shape, (1.4, 1.8), alpha=0.5)
    g_sp = kf.build(g_lr.astype(float), np.full(ssv_hr.shape, np.nan),
                    np.zeros(ssv_hr.shape, dtype=bool), lon, lat, LON, LAT,
                    upscale=up, bathy_15s=bathy, n_iter=6, smooth_sigma=4.0)
    x1, x2, x3 = MultiScaleKF(ssv_hr, hp, alpha=0.5).run(
        g_lr.astype(float), np.zeros_like(ssv_hr),
        np.zeros_like(ssv_hr, dtype=bool), n_iter=6)

    def mk(field, pct):
        b = bandpass(field, dx, dy, 5e3, 20e3)
        return filter_lineaments(lineament_mask(b, dx, dy, pct=pct)
                                 & (bathy < 0), min_area=20, min_elong=1.8)

    m_sp = mk(g_sp, 70)
    m_x3 = mk(x3, 50)
    masks = {
        "spatial70": m_sp,
        "x3_50": m_x3,
        "union": m_sp | m_x3,
        "inter": m_sp & m_x3,
    }

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
    folds = np.array_split(idx, 5)

    print(f"{'mask':10s} {'mean':>6s} {'std':>5s}")
    for name, m in masks.items():
        vals = []
        for fold in folds:
            tx, ty = densify([lines[i] for i in fold], 0.5)
            vals.append(fault_match(m, tx, ty, lon, lat, radius_km=5)["recall"] * 100)
        print(f"{name:10s} {np.mean(vals):6.1f} {np.std(vals):5.1f}")


if __name__ == "__main__":
    main()
