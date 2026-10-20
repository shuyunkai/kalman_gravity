"""围绕 KF 的多种尺度分解场景，统一用 GEM 断层吻合率评估，找最优方案。

场景：
  S1 带通 10–50 km（现方案基线）
  S2 多尺度 KF（σ=16/4/1）的不同分量组合
  S3 多尺度 KF（σ=32/8/2）
  S4 单次 KF 的 σ 扫描，相邻场相减得到尺度带
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from three_method_compare import load_region_data
from kf_novel import BathyPriorKF
from multiscale_kf import MultiScaleKF
from scale_decomposition import bandpass, lineament_mask, filter_lineaments
from fault_validation import load_gem_faults, densify, fault_match

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real" / "three_parts"
LON, LAT = (141.0, 149.0), (32.0, 40.0)
CACHE = OUT / "kf_inputs.npz"


def get_data():
    if CACHE.exists():
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}
    d = load_region_data(LON, LAT, val_cruise="MR22-04_gravity_dmo", upscale=4)
    keep = {k: d[k] for k in ("gravity_lr", "ssv_hr", "bathy_15s", "ship_grav",
                              "train_mask", "lon_hr", "lat_hr")}
    np.savez_compressed(CACHE, **keep)
    return keep


def eval_field(f, ocean, flon, flat, lon, lat, dx, dy, pct=84):
    mask = filter_lineaments(lineament_mask(f, dx, dy, pct=pct) & ocean,
                             min_area=60, min_elong=2.2)
    st = fault_match(mask, flon, flat, lon, lat, radius_km=50)
    st20 = fault_match(mask, flon, flat, lon, lat, radius_km=20)
    return mask.sum(), st["recall"] * 100, st["recall_random"] * 100, \
        st["precision"] * 100, st20["recall"] * 100


def main():
    d = get_data()
    lon, lat = d["lon_hr"], d["lat_hr"]
    bathy = d["bathy_15s"].astype(float)
    dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = float(lat[1] - lat[0]) * 111320
    ocean = bathy < 0

    flines = load_gem_faults(OUT / "gem_active_faults.geojson",
                             lon.min(), lon.max(), lat.min(), lat.max())
    flon, flat = densify(flines, 2.0)

    kf = BathyPriorKF(d["ssv_hr"].shape, d["gravity_lr"].shape, (8, 8), alpha=0.5)
    fields = {}
    for s in (2.0, 4.0, 8.0, 16.0, 32.0):
        fields[s] = kf.build(d["gravity_lr"].astype(float), d["ship_grav"],
                             d["train_mask"], lon, lat, LON, LAT,
                             upscale=4, bathy_15s=bathy, n_iter=6,
                             smooth_sigma=s)
        print("KF sigma", s, "done")

    rows = []

    def add(name, f):
        n, r50, rnd50, p50, r20 = eval_field(f, ocean, flon, flat, lon, lat, dx, dy)
        rows.append(dict(scenario=name, lineaments=int(n), recall50=r50,
                         rand50=rnd50, precision50=p50, recall20=r20))
        print(f"{name:28s} n={n:7,d} recall50={r50:5.1f} (rand {rnd50:4.1f}) "
              f"prec50={p50:5.1f} recall20={r20:4.1f}")

    add("S1 bandpass 10-50", bandpass(fields[4.0], dx, dy, 10e3, 50e3))

    for tag, sig in (("S2 ms 16/4/1", (16.0, 4.0, 1.0)),
                     ("S3 ms 32/8/2", (32.0, 8.0, 2.0))):
        m = MultiScaleKF(d["ssv_hr"].astype(float),
                         _terrain_hp(bathy), sigmas=sig,
                         q=(sig[0], sig[1], sig[2]))
        x1, x2, x3 = m.run(d["gravity_lr"].astype(float), d["ship_grav"],
                           d["train_mask"], n_iter=6)
        add(f"{tag} x1+x2", x1 + x2)
        add(f"{tag} x2", x2)
        add(f"{tag} x2+x3", x2 + x3)
        add(f"{tag} sum", x1 + x2 + x3)

    for a, b in ((2.0, 4.0), (4.0, 8.0), (8.0, 16.0), (16.0, 32.0)):
        add(f"S4 KF band {a:.0f}-{b:.0f}", fields[a] - fields[b])

    rows.sort(key=lambda r: r["recall50"], reverse=True)
    with open(OUT / "kf_scenarios.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("\nTOP:")
    for r in rows[:5]:
        print(r)


def _terrain_hp(bathy):
    from scipy.ndimage import gaussian_filter
    c = np.where(bathy < 0, 0.0688, 0.1120)
    boug = c * bathy
    return boug - gaussian_filter(boug, 12.0)


if __name__ == "__main__":
    main()
