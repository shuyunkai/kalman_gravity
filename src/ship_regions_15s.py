"""有船测的三个 15″ 区域：SSV / KF / BOL 场与留出航次 RMSE。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from three_method_compare import load_region_data
from kf_novel import BathyPriorKF

REGIONS = {
    "D_Nankai": dict(lon=(124.0, 132.0), lat=(24.0, 31.0),
                     val="YK14-14_gravity_dmo"),
    "F_PhilSea": dict(lon=(139.0, 152.0), lat=(13.0, 26.0),
                      val="YK18-08_gravity_dmo"),
    "G_Japan": dict(lon=(132.0, 152.0), lat=(26.0, 42.0),
                    val="MR22-04_gravity_dmo"),
}
OUT = Path(__file__).parent.parent / "output_real" / "three_parts" / "ship_regions"
OUT.mkdir(parents=True, exist_ok=True)


def rmse_at(model, lon, lat, pts):
    ix = np.clip(np.searchsorted(lon, pts[:, 0]), 0, len(lon) - 1)
    iy = np.clip(np.searchsorted(lat, pts[:, 1]), 0, len(lat) - 1)
    d = model[iy, ix] - pts[:, 2]
    return float(np.sqrt(np.mean(d ** 2)))


def run(name, cfg):
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

    kf = BathyPriorKF(*shape, (8.0, 8.0), ship_noise_std=10.0, alpha=0.5)
    g_kf = kf.build(**base)
    bol = BathyPriorKF(*shape, (8.0, 8.0), ship_noise_std=10.0, alpha=0.5,
                       p0_std=4.0, q_std=4.0, adaptive_ship=False)
    g_bol = bol.build(**base, localize=True, adaptive=True, corr_px=28.0)

    for tag, g in (("ssv", d["ssv_hr"]), ("kf", g_kf), ("bol", g_bol)):
        print(f"  {tag}: {rmse_at(g, lon, lat, pts):.4f}")
    np.savez_compressed(OUT / f"{name}.npz", lon=lon, lat=lat,
                        ssv=d["ssv_hr"].astype(np.float32),
                        kf=g_kf.astype(np.float32),
                        bol=g_bol.astype(np.float32),
                        bathy=d["bathy_15s"].astype(np.float32))
    print("  saved", OUT / f"{name}.npz")


if __name__ == "__main__":
    for k, v in REGIONS.items():
        run(k, v)
