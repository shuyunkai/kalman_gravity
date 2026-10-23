"""用扰动 KF 成员估计集合相关函数与相关长度。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from three_method_compare import load_region_data
from kf_novel import BathyPriorKF

LON = (141.0, 149.0)
LAT = (32.0, 40.0)
PERT = [(0.35, 1.8, 3.0), (0.45, 2.2, 3.5), (0.50, 2.5, 4.0),
        (0.55, 2.8, 4.5), (0.60, 3.2, 5.0), (0.70, 3.6, 5.5)]


def main():
    d = load_region_data(LON, LAT, val_cruise="MR22-04_gravity_dmo",
                         upscale=4)
    lon, lat = d["lon_hr"], d["lat_hr"]
    base = dict(gravity_lr=d["gravity_lr"].astype(float),
                ship_15s=d["ship_grav"], train_mask=d["train_mask"],
                lon_hr=lon, lat_hr=lat, lon_range=LON, lat_range=LAT,
                upscale=4, bathy_15s=d["bathy_15s"], n_iter=6,
                smooth_sigma=4.0)
    shape = (d["ssv_hr"].shape, d["gravity_lr"].shape)
    ens = []
    for a, q, rs in PERT:
        m = BathyPriorKF(*shape, (8.0, 8.0), ship_noise_std=rs, alpha=a)
        g = m.build(**base)
        ens.append(g.astype(np.float32))
    E = np.stack(ens)                     # (N, ny, nx)
    A = E - E.mean(axis=0, keepdims=True)
    print("ensemble", E.shape, "spread", float(A.std()))

    rng = np.random.default_rng(0)
    yy = rng.integers(12, A.shape[1] - 12, 4000)
    xx = rng.integers(12, A.shape[2] - 12, 4000)
    rhos = []
    for lag in range(1, 11):
        vals = []
        for dy, dx in ((lag, 0), (0, lag)):
            a = A[:, yy, xx]
            b = A[:, yy + dy, xx + dx]
            num = np.mean(a * b, axis=0)
            den = np.sqrt(np.mean(a * a, axis=0) * np.mean(b * b, axis=0))
            vals.append(num / np.maximum(den, 1e-12))
        rhos.append(float(np.nanmean(vals)))
    rhos = np.array(rhos)
    print("rho(d):", np.round(rhos, 3))
    # Gaussian 拟合: rho = exp(-d^2/(2 l^2))
    dist = np.arange(1, 11)
    ok = rhos > 0.05
    l2 = -dist[ok] ** 2 / (2 * np.log(np.maximum(rhos[ok], 1e-6)))
    print("l_eff px (median)", float(np.median(l2)), "km",
          float(np.median(l2)) * 0.46)


if __name__ == "__main__":
    main()
