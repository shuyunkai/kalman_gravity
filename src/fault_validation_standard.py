"""按公开文献常用缓冲半径（2/5/10 km）验证断层吻合率。"""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent))

from multiscale_kf import MultiScaleKF
from scale_decomposition import bandpass, lineament_mask, filter_lineaments
from fault_validation import load_gem_faults, densify, fault_match

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real" / "three_parts"


def main():
    d = np.load(OUT / "kf_inputs.npz")
    bathy = d["bathy_15s"].astype(float)
    lon, lat = d["lon_hr"], d["lat_hr"]
    dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = float(lat[1] - lat[0]) * 111320
    ocean = bathy < 0

    c = np.where(bathy < 0, 0.0688, 0.1120)
    boug = c * bathy
    hp = boug - gaussian_filter(boug, 12.0)
    m = MultiScaleKF(d["ssv_hr"].astype(float), hp, sigmas=(16., 4., 1.),
                     q=(16., 4., 1.))
    x1, x2, x3 = m.run(d["gravity_lr"].astype(float), d["ship_grav"],
                       d["train_mask"], n_iter=6)
    f_kf = x1 + x2 + x3

    flines = load_gem_faults(OUT / "gem_active_faults.geojson",
                             lon.min(), lon.max(), lat.min(), lat.max())
    flon, flat = densify(flines, 1.0)

    fields = {
        "KF 多尺度求和": f_kf,
        "带通 10–50 km": bandpass(f_kf, dx, dy, 10e3, 50e3),
    }
    print(f"GEM lines {len(flines)}, points {len(flon):,}")
    for name, f in fields.items():
        mask = filter_lineaments(lineament_mask(f, dx, dy, pct=84) & ocean,
                                 min_area=60, min_elong=2.2)
        print(f"\n{name}: lineaments={mask.sum():,}")
        for r in (2, 5, 10, 20, 50):
            st = fault_match(mask, flon, flat, lon, lat, radius_km=r)
            print(f"  r={r:2d} km: 覆盖 {st['recall']*100:5.1f}% "
                  f"(随机 {st['recall_random']*100:4.1f}%) | "
                  f"命中 {st['precision']*100:5.1f}% "
                  f"(随机 {st['precision_random']*100:4.1f}%)")


if __name__ == "__main__":
    main()
