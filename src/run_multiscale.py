"""多尺度 KF 原型：跑测试区，输出三个尺度分量并做断层吻合率对比。"""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent))

from three_method_compare import load_region_data
from multiscale_kf import MultiScaleKF
from scale_decomposition import lineament_mask, filter_lineaments, bandpass, thg
from fault_validation import load_gem_faults, densify, fault_match

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real" / "three_parts"
LON, LAT = (141.0, 149.0), (32.0, 40.0)


def main():
    data = load_region_data(LON, LAT, val_cruise="MR22-04_gravity_dmo", upscale=4)
    b = data["bathy_15s"].astype(float)
    c = np.where(b < 0, 0.0688, 0.1120)
    boug = c * b
    hp = boug - gaussian_filter(boug, 12.0)

    mskf = MultiScaleKF(data["ssv_hr"].astype(float), hp, alpha=0.5)
    x1, x2, x3 = mskf.run(data["gravity_lr"].astype(float),
                          data["ship_grav"], data["train_mask"], n_iter=6)
    np.savez_compressed(OUT / "multiscale_kf.npz", x1=x1, x2=x2, x3=x3,
                        lon=data["lon_hr"], lat=data["lat_hr"])
    print("saved multiscale_kf.npz", x1.shape)

    lon, lat = data["lon_hr"], data["lat_hr"]
    dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = float(lat[1] - lat[0]) * 111320
    ocean = b < 0
    flines = load_gem_faults(OUT / "gem_active_faults.geojson",
                             lon.min(), lon.max(), lat.min(), lat.max())
    flon, flat = densify(flines, 2.0)

    for name, f in (("x2 中波", x2), ("x3 细波", x3), ("x1+x2", x1 + x2)):
        mask = filter_lineaments(lineament_mask(f, dx, dy, pct=84) & ocean,
                                 min_area=60, min_elong=2.2)
        st = fault_match(mask, flon, flat, lon, lat, radius_km=50)
        print(f"{name}: lineaments={mask.sum():,} recall50={st['recall']*100:.1f} "
              f"(rand {st['recall_random']*100:.1f}) precision50={st['precision']*100:.1f}")

    # 对照：带通 10–50 km
    g15 = x1 + x2 + x3
    bp = bandpass(g15, dx, dy, 10e3, 50e3)
    mask = filter_lineaments(lineament_mask(bp, dx, dy, pct=84) & ocean,
                             min_area=60, min_elong=2.2)
    st = fault_match(mask, flon, flat, lon, lat, radius_km=50)
    print(f"bandpass 10-50: lineaments={mask.sum():,} recall50={st['recall']*100:.1f} "
          f"(rand {st['recall_random']*100:.1f})")


if __name__ == "__main__":
    main()
