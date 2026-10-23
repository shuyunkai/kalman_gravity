"""BOL-AKF 训练/验证网格：P0、Q、相关长度；自适应只调 R_sat。"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from three_method_compare import load_region_data
from kf_novel import BathyPriorKF

LON = (141.0, 149.0)
LAT = (32.0, 40.0)


def rmse_at(model, lon, lat, pts):
    ix = np.clip(np.searchsorted(lon, pts[:, 0]), 0, len(lon) - 1)
    iy = np.clip(np.searchsorted(lat, pts[:, 1]), 0, len(lat) - 1)
    d = model[iy, ix] - pts[:, 2]
    return float(np.sqrt(np.mean(d ** 2)))


def main():
    data = load_region_data(LON, LAT, val_cruise="MR22-04_gravity_dmo",
                            upscale=4)
    lon, lat = data["lon_hr"], data["lat_hr"]
    pts = np.asarray(data["val_points"], float)
    base = dict(gravity_lr=data["gravity_lr"].astype(float),
                ship_15s=data["ship_grav"], train_mask=data["train_mask"],
                lon_hr=lon, lat_hr=lat, lon_range=LON, lat_range=LAT,
                upscale=4, bathy_15s=data["bathy_15s"], n_iter=6,
                smooth_sigma=4.0)
    shape = (data["ssv_hr"].shape, data["gravity_lr"].shape)

    kf = BathyPriorKF(*shape, (8.0, 8.0), ship_noise_std=10.0, alpha=0.5)
    print("baseline KF", rmse_at(kf.build(**base), lon, lat, pts))

    best = None
    for p0, q, cp in itertools.product((2.0, 4.0), (2.0, 4.0), (2.0, 6.0)):
        m = BathyPriorKF(*shape, (8.0, 8.0), ship_noise_std=10.0, alpha=0.5,
                         p0_std=p0, q_std=q, adaptive_ship=False)
        g = m.build(**base, localize=True, adaptive=True, corr_px=cp)
        e = rmse_at(g, lon, lat, pts)
        print(f"p0={p0} q={q} cp={cp}: {e:.4f}")
        if best is None or e < best[0]:
            best = (e, p0, q, cp)
    print("best", best)


if __name__ == "__main__":
    main()
