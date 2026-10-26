"""测试两个候选区域，寻找相干保持 KF 增益更明显的验证区。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from three_method_compare import load_region_data
from kf_novel import BathyPriorKF
from aniso_kf_test import build as build_ckf

CANDIDATES = {
    "H_Kuroshio": dict(lon=(134.0, 141.0), lat=(31.0, 36.0),
                       val="YK18-14_gravity_dmo"),
    "I_NWPacific": dict(lon=(145.0, 160.0), lat=(34.0, 42.0),
                        val="KR14-07_gravity_dmo"),
}


def rmse_at(model, lon, lat, pts):
    ix = np.clip(np.searchsorted(lon, pts[:, 0]), 0, len(lon) - 1)
    iy = np.clip(np.searchsorted(lat, pts[:, 1]), 0, len(lat) - 1)
    return float(np.sqrt(np.mean((model[iy, ix] - pts[:, 2]) ** 2)))


for name, cfg in CANDIDATES.items():
    print("===", name)
    d = load_region_data(cfg["lon"], cfg["lat"], val_cruise=cfg["val"], upscale=4)
    lon, lat = d["lon_hr"], d["lat_hr"]
    pts = np.asarray(d["val_points"], float)
    kf = BathyPriorKF(d["ssv_hr"].shape, d["gravity_lr"].shape, (5.0, 7.0),
                      ship_noise_std=10.0, alpha=0.5)
    g_kf = kf.build(gravity_lr=d["gravity_lr"].astype(float),
                    ship_15s=d["ship_grav"], train_mask=d["train_mask"],
                    lon_hr=lon, lat_hr=lat, lon_range=cfg["lon"],
                    lat_range=cfg["lat"], upscale=4, bathy_15s=d["bathy_15s"],
                    n_iter=6, smooth_sigma=4.0)
    g_ckf = build_ckf(d["gravity_lr"].astype(float), d["bathy_15s"],
                      lon, lat, w_keep=1.0, p0=2.0, n_iter=6,
                      ship=d["ship_grav"], train=d["train_mask"])
    s = rmse_at(d["ssv_hr"], lon, lat, pts)
    k = rmse_at(g_kf, lon, lat, pts)
    c = rmse_at(g_ckf, lon, lat, pts)
    print(f"  n={len(pts)} SSV={s:.3f} KF={k:.3f} CKF={c:.3f} "
          f"vs1={100*(1-c/s):.2f}% vsKF={100*(1-c/k):.2f}%")
