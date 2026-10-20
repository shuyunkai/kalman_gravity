"""断裂候选与 GEM 活动断层的吻合率 / 走向一致性（可复现实验）。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from scale_decomposition import bandpass, lineament_mask, filter_lineaments
from fault_validation import (load_gem_faults, densify, fault_match,
                              lineament_segments, _xy)

ROOT = Path(__file__).parent.parent
NPZ = ROOT / "output_real" / "test_field_15s.npz"
GEM = ROOT / "output_real" / "three_parts" / "gem_active_faults.geojson"


def main():
    d = np.load(NPZ)
    g = d["g15"].astype(float)
    bathy = d["bathy"].astype(float)
    lon, lat = d["lon"], d["lat"]
    dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = float(lat[1] - lat[0]) * 111320

    flines = load_gem_faults(GEM, lon.min(), lon.max(), lat.min(), lat.max())
    flon, flat = densify(flines, 2.0)
    print(f"GEM lines {len(flines)}, points {len(flon):,}")

    for lab, (lo, hi) in {"10–50 km": (10e3, 50e3), "50–100 km": (50e3, 100e3)}.items():
        b = bandpass(g, dx, dy, lo, hi)
        mask = filter_lineaments(lineament_mask(b, dx, dy, pct=84) & (bathy < 0),
                                 min_area=60, min_elong=2.2)
        for r in (20, 30, 50):
            st = fault_match(mask, flon, flat, lon, lat, radius_km=r)
            print(f"{lab} r={r}: recall={st['recall']*100:.1f} "
                  f"(rand {st['recall_random']*100:.1f}) "
                  f"precision={st['precision']*100:.1f} "
                  f"(rand {st['precision_random']*100:.1f})")

    # 局部构造方向（结构张量）与断层走向比较：
    # 注意不能用“最近线段的方位”，稀疏线段会互相匹配出伪一致。
    from scipy.ndimage import gaussian_filter

    b = bandpass(g, dx, dy, 10e3, 50e3)
    gy, gx = np.gradient(b, dy, dx)
    s = 8
    Jxx = gaussian_filter(gx * gx, s)
    Jyy = gaussian_filter(gy * gy, s)
    Jxy = gaussian_filter(gx * gy, s)
    theta = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)
    line_dir = (np.degrees(theta) + 90.0) % 180.0

    def sample(x, y):
        i = np.clip(np.searchsorted(lat, y) - 1, 0, len(lat) - 1)
        j = np.clip(np.searchsorted(lon, x) - 1, 0, len(lon) - 1)
        return line_dir[i, j]

    diffs = []
    for arr in flines:
        for (x0, y0), (x1, y1) in zip(arr[:-1], arr[1:]):
            strike = np.degrees(np.arctan2(y1 - y0,
                                           (x1 - x0) * np.cos(np.radians(y0)))) % 180
            ld = sample((x0 + x1) / 2, (y0 + y1) / 2)
            diff = abs(strike - ld)
            diffs.append(min(diff, 180 - diff))
    diffs = np.array(diffs)
    print("fault segments", len(diffs))
    for tol in (15, 20, 30):
        print(f"local orientation within {tol} deg: "
              f"{(diffs <= tol).mean()*100:.1f}% "
              f"(random {2*tol/180*100:.1f}%)")
    print("median diff", np.median(diffs))


if __name__ == "__main__":
    main()
