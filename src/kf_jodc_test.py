"""把 JODC 船测重力并入 KF，验证对留出航次 MR22-04 的影响。"""
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).parent))

from three_method_compare import load_region_data
from kf_novel import BathyPriorKF

ROOT = Path(__file__).parent.parent


def build_ship(data, use_jodc=False):
    ship = data["ship_grav"].astype(float).copy()
    mask = data["train_mask"].copy()
    if use_jodc:
        d = np.load(ROOT / "data" / "ship_gravity" / "jodc_fa.npz")
        lon, lat = d["lon"], d["lat"]
        grav = d["fa"]
        ok = np.isfinite(grav)
        ok &= (lon >= data["lon_hr"].min()) & (lon <= data["lon_hr"].max())
        ok &= (lat >= data["lat_hr"].min()) & (lat <= data["lat_hr"].max())
        ix = np.clip(np.searchsorted(data["lon_hr"], lon[ok]), 0, data["nx_hr"] - 1)
        iy = np.clip(np.searchsorted(data["lat_hr"], lat[ok]), 0, data["ny_hr"] - 1)
        add = np.where(~mask[iy, ix])[0]
        ship[iy[add], ix[add]] = grav[ok][add]
        mask[iy, ix] = True
        print("JODC added cells", len(add))
    return ship, mask


def main():
    data = load_region_data((141, 149), (32, 40),
                            val_cruise="MR22-04_gravity_dmo", upscale=4)
    vp = data["val_points"]
    vlon, vlat, vgrav = vp[:, 0], vp[:, 1], vp[:, 2]

    for tag, use, noise in (("jp only", False, 1.0),
                            ("combined n1", True, 1.0),
                            ("combined n5", True, 5.0),
                            ("combined n10", True, 10.0)):
        ship, mask = build_ship(data, use)
        kf = BathyPriorKF(data["ssv_hr"].shape, data["gravity_lr"].shape,
                          (8.0, 8.0), ship_noise_std=noise, alpha=0.5)
        g = kf.build(data["gravity_lr"].astype(float), ship, mask,
                     data["lon_hr"], data["lat_hr"], (141, 149), (32, 40),
                     upscale=4, bathy_15s=data["bathy_15s"], n_iter=6,
                     smooth_sigma=4.0)
        f = RegularGridInterpolator((data["lat_hr"], data["lon_hr"]), g,
                                    bounds_error=False, fill_value=np.nan)
        pred = f(np.column_stack([vlat, vlon]))
        m = np.isfinite(pred)
        rmse = np.sqrt(np.mean((pred[m] - vgrav[m]) ** 2))
        bias = np.mean(pred[m] - vgrav[m])
        print(f"{tag}: n={m.sum():,} RMSE={rmse:.3f} mGal bias={bias:+.3f}")


if __name__ == "__main__":
    main()
