"""对比 SSV / 当前 KF / BOL-AKF 在留出航次上的 RMSE。"""
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
    return float(np.sqrt(np.mean(d ** 2))), float(np.mean(d))


def main():
    data = load_region_data(LON, LAT, val_cruise="MR22-04_gravity_dmo", upscale=4)
    lon, lat = data["lon_hr"], data["lat_hr"]
    pts = np.asarray(data["val_points"], float)
    print("val points", pts.shape)

    base = dict(gravity_lr=data["gravity_lr"].astype(float),
                ship_15s=data["ship_grav"], train_mask=data["train_mask"],
                lon_hr=lon, lat_hr=lat, lon_range=LON, lat_range=LAT,
                upscale=4, bathy_15s=data["bathy_15s"], n_iter=6,
                smooth_sigma=4.0)
    ss = data["ssv_hr"]
    print("SSV", rmse_at(ss, lon, lat, pts))

    kf = BathyPriorKF(data["ssv_hr"].shape, data["gravity_lr"].shape,
                      (8.0, 8.0), ship_noise_std=1.0, alpha=0.5)
    g0 = kf.build(**base)
    print("KF", rmse_at(g0, lon, lat, pts))

    for cp in (1.0, 2.0, 6.0):
        bol = BathyPriorKF(data["ssv_hr"].shape, data["gravity_lr"].shape,
                           (8.0, 8.0), ship_noise_std=1.0, alpha=0.5)
        g1 = bol.build(**base, localize=True, adaptive=False, corr_px=cp)
        print(f"BOL cp={cp}", rmse_at(g1, lon, lat, pts))


if __name__ == "__main__":
    main()
