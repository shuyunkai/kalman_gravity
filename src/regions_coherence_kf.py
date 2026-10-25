"""三个区域：SSV / 普通 KF / 相干保持 KF 的留出验证。"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from three_method_compare import load_region_data
from kf_novel import BathyPriorKF
from aniso_kf_test import build as build_aniso

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real" / "three_parts" / "newdeck"

REGIONS = {
    "D_Nankai": dict(lon=(124.0, 132.0), lat=(24.0, 31.0),
                     val="YK14-14_gravity_dmo"),
    "F_PhilSea": dict(lon=(139.0, 152.0), lat=(13.0, 26.0),
                      val="YK18-08_gravity_dmo"),
    "G_Japan": dict(lon=(132.0, 152.0), lat=(26.0, 42.0),
                    val="MR22-04_gravity_dmo"),
}


def rmse_at(model, lon, lat, pts):
    ix = np.clip(np.searchsorted(lon, pts[:, 0]), 0, len(lon) - 1)
    iy = np.clip(np.searchsorted(lat, pts[:, 1]), 0, len(lat) - 1)
    d = model[iy, ix] - pts[:, 2]
    return float(np.sqrt(np.mean(d ** 2)))


def main():
    res = {}
    for name, cfg in REGIONS.items():
        print("===", name)
        d = load_region_data(cfg["lon"], cfg["lat"], val_cruise=cfg["val"],
                             upscale=4)
        lon, lat = d["lon_hr"], d["lat_hr"]
        pts = np.asarray(d["val_points"], float)
        base = dict(gravity_lr=d["gravity_lr"].astype(float),
                    ship_15s=d["ship_grav"], train_mask=d["train_mask"],
                    lon_hr=lon, lat_hr=lat, lon_range=cfg["lon"],
                    lat_range=cfg["lat"], upscale=4, bathy_15s=d["bathy_15s"],
                    n_iter=6, smooth_sigma=4.0)
        shape = (d["ssv_hr"].shape, d["gravity_lr"].shape)

        ssv = rmse_at(d["ssv_hr"], lon, lat, pts)
        kf = BathyPriorKF(*shape, (8.0, 8.0), ship_noise_std=10.0, alpha=0.5)
        g_kf = kf.build(**base)
        e_kf = rmse_at(g_kf, lon, lat, pts)
        g_c = build_aniso(d["gravity_lr"].astype(float), d["bathy_15s"],
                          lon, lat, w_keep=1.0, p0=2.0, n_iter=6,
                          ship=d["ship_grav"], train=d["train_mask"])
        e_c = rmse_at(g_c, lon, lat, pts)
        print(f"  SSV {ssv:.4f}  KF {e_kf:.4f}  Coherence {e_c:.4f}")
        res[name] = dict(ssv=ssv, kf=e_kf, ckf=e_c,
                         n=int(len(pts)))
    (OUT / "regions_coherence.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
