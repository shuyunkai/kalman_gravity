"""保存 G 区 SSV / 标准 KF / 相干保持 KF 场，供验证大图使用。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from three_method_compare import load_region_data
from kf_novel import BathyPriorKF
from aniso_kf_test import build as build_ckf

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"

LON, LAT = (132.0, 152.0), (26.0, 42.0)


def main():
    d = load_region_data(LON, LAT, val_cruise="MR22-04_gravity_dmo", upscale=4)
    base = dict(gravity_lr=d["gravity_lr"].astype(float),
                ship_15s=d["ship_grav"], train_mask=d["train_mask"],
                lon_hr=d["lon_hr"], lat_hr=d["lat_hr"], lon_range=LON,
                lat_range=LAT, upscale=4, bathy_15s=d["bathy_15s"],
                n_iter=6, smooth_sigma=4.0)
    kf = BathyPriorKF(d["ssv_hr"].shape, d["gravity_lr"].shape, (16.0, 20.0),
                      ship_noise_std=10.0, alpha=0.5)
    g_kf = kf.build(**base)
    g_ckf = build_ckf(d["gravity_lr"].astype(float), d["bathy_15s"],
                      d["lon_hr"], d["lat_hr"], w_keep=1.0, p0=2.0, n_iter=6,
                      ship=d["ship_grav"], train=d["train_mask"])
    np.savez_compressed(OUT / "validation_G_fields.npz",
                        lon=d["lon_hr"], lat=d["lat_hr"],
                        ssv=d["ssv_hr"].astype(np.float32),
                        kf=g_kf.astype(np.float32),
                        ckf=g_ckf.astype(np.float32),
                        val=np.asarray(d["val_points"], np.float32))
    print("saved", OUT / "validation_G_fields.npz")


if __name__ == "__main__":
    main()
