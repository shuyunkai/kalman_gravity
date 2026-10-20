"""断裂提取参数搜索：KF 场 + 短频带 + 阈值/面积，训练/检验分段评估。

目标：在“海域断层编目”窗口里，找到能达到区域研究量级（2–10 km 缓冲下
覆盖 20–50%）的参数，同时精度显著高于随机基线。
"""
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
from fault_validation import densify, fault_match

OUT = Path(__file__).parent.parent / "output_real" / "three_parts"


def build_field(LON, LAT):
    g_lr, _, _, _ = load_ssv_region(LON, LAT)
    ny, nx = g_lr.shape
    up = 4
    lon = cell_centers(LON[0], LON[1], nx * up)
    lat = cell_centers(LAT[0], LAT[1], ny * up)
    bathy = load_bathy_15s(LON, LAT, lon, lat)
    ssv_hr = sp_zoom(g_lr, (up, up), order=1)
    ship = np.full(ssv_hr.shape, np.nan)
    m0 = np.zeros(ssv_hr.shape, dtype=bool)
    kf = BathyPriorKF(ssv_hr.shape, g_lr.shape,
                      ((LAT[1] - LAT[0]), (LON[1] - LON[0])), alpha=0.5)
    g = kf.build(g_lr.astype(float), ship, m0, lon, lat, LON, LAT,
                 upscale=up, bathy_15s=bathy, n_iter=6, smooth_sigma=4.0)
    return g, bathy, lon, lat


def load_lines(path, LON, LAT):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
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
    return lines


def split_lines(lines, frac=0.5):
    idx = np.arange(len(lines))
    rng = np.random.default_rng(7)
    rng.shuffle(idx)
    k = int(len(lines) * frac)
    return [lines[i] for i in idx[:k]], [lines[i] for i in idx[k:]]


def eval_mask(mask, lines, lon, lat, radii=(2, 5, 10)):
    x, y = densify(lines, 0.5)
    out = []
    for r in radii:
        st = fault_match(mask, x, y, lon, lat, radius_km=r)
        out.append((r, st["recall"] * 100, st["recall_random"] * 100,
                    st["precision"] * 100, st["precision_random"] * 100))
    return out


def main():
    cases = [
        ("CA", (-124.5, -121.5), (33.0, 35.5), OUT / "ca_faults.json"),
        ("GM9", (170.3, 172.1), (-46.4, -45.0), OUT / "nz_gm9_offshore.json"),
    ]
    for name, LON, LAT, fpath in cases:
        g, bathy, lon, lat = build_field(LON, LAT)
        dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
        dy = float(lat[1] - lat[0]) * 111320
        lines = load_lines(fpath, LON, LAT)
        tr, te = split_lines(lines, 0.5)
        print(f"\n=== {name}: lines {len(lines)} (train {len(tr)}, test {len(te)}) ===")

        best = None
        for lo, hi in ((5e3, 20e3), (10e3, 30e3)):
            b = bandpass(g, dx, dy, lo, hi)
            for pct in (60, 70, 78, 84):
                mask = filter_lineaments(lineament_mask(b, dx, dy, pct=pct)
                                         & (bathy < 0), min_area=20,
                                         min_elong=1.8)
                rows = eval_mask(mask, tr, lon, lat)
                r5 = next(r for r in rows if r[0] == 5)
                score = r5[1] + r5[3] * 0.5
                print(f"band {lo/1e3:.0f}-{hi/1e3:.0f}, pct={pct}: "
                      f"n={mask.sum():5d} train5km cov={r5[1]:5.1f}% "
                      f"(rand {r5[2]:4.1f}) prec={r5[3]:5.1f}%")
                if best is None or score > best[0]:
                    best = (score, lo, hi, pct, mask)
        _, lo, hi, pct, mask = best
        print(f"BEST {name}: band {lo/1e3:.0f}-{hi/1e3:.0f}, pct={pct}")
        for r, cov, rc, pre, prc in eval_mask(mask, te, lon, lat):
            print(f"  test r={r:2d}: coverage {cov:5.1f}% (rand {rc:4.1f}) "
                  f"precision {pre:5.1f}% (rand {prc:4.1f})")


if __name__ == "__main__":
    main()
