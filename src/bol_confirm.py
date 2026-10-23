"""在另外两条留出航次上确认 BOL 配置。"""
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


def run(cruise):
    data = load_region_data(LON, LAT, val_cruise=cruise, upscale=4)
    lon, lat = data["lon_hr"], data["lat_hr"]
    pts = np.asarray(data["val_points"], float)
    base = dict(gravity_lr=data["gravity_lr"].astype(float),
                ship_15s=data["ship_grav"], train_mask=data["train_mask"],
                lon_hr=lon, lat_hr=lat, lon_range=LON, lat_range=LAT,
                upscale=4, bathy_15s=data["bathy_15s"], n_iter=6,
                smooth_sigma=4.0)
    shape = (data["ssv_hr"].shape, data["gravity_lr"].shape)
    kf = BathyPriorKF(*shape, (8.0, 8.0), ship_noise_std=10.0, alpha=0.5)
    e0 = rmse_at(kf.build(**base), lon, lat, pts)
    bol = BathyPriorKF(*shape, (8.0, 8.0), ship_noise_std=10.0, alpha=0.5,
                       p0_std=4.0, q_std=4.0, adaptive_ship=False)
    e1 = rmse_at(bol.build(**base, localize=True, adaptive=True, corr_px=6.0),
                 lon, lat, pts)
    print(f"{cruise}: KF {e0:.4f}  BOL {e1:.4f}  diff {e1-e0:+.4f}")


if __name__ == "__main__":
    for c in ("MR23-02_gravity_dmo", "MR21-02_gravity_dmo"):
        run(c)
