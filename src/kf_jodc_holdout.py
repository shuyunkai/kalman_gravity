"""留出一个 JODC 航次，检验其余 JODC + jp_data 对 KF 的增益。"""
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).parent))

from three_method_compare import load_region_data
from kf_novel import BathyPriorKF

ROOT = Path(__file__).parent.parent
HOLD = "HM921128"


def main():
    data = load_region_data((141, 149), (32, 40),
                            val_cruise="MR22-04_gravity_dmo", upscale=4)
    d = np.load(ROOT / "data" / "ship_gravity" / "jodc_fa.npz")
    lon, lat, fa, voy = d["lon"], d["lat"], d["fa"], d["voyage"]
    ssvp = RegularGridInterpolator(
        (data["lat_hr"], data["lon_hr"]), data["ssv_hr"],
        bounds_error=False, fill_value=np.nan)(np.column_stack([lat, lon]))
    fa = fa.copy()
    for v in np.unique(voy):
        sel = (voy == v) & np.isfinite(ssvp)
        if sel.sum() > 50:
            fa[sel] -= np.median(fa[sel] - ssvp[sel])

    hold = voy == HOLD
    vlon, vlat, vfa = lon[hold], lat[hold], fa[hold]
    print("holdout points", hold.sum())

    base_ship = data["ship_grav"].astype(float).copy()
    base_mask = data["train_mask"].copy()

    add = (~hold) & np.isfinite(fa)
    add &= (lon >= data["lon_hr"].min()) & (lon <= data["lon_hr"].max())
    add &= (lat >= data["lat_hr"].min()) & (lat <= data["lat_hr"].max())
    ix = np.clip(np.searchsorted(data["lon_hr"], lon[add]), 0, data["nx_hr"] - 1)
    iy = np.clip(np.searchsorted(data["lat_hr"], lat[add]), 0, data["ny_hr"] - 1)

    comb_ship = base_ship.copy()
    comb_mask = base_mask.copy()
    free = ~comb_mask[iy, ix]
    comb_ship[iy[free], ix[free]] = fa[add][free]
    comb_mask[iy, ix] = True
    print("JODC added cells", int(free.sum()))

    for tag, ship, mask, noise in (
            ("jp_data only", base_ship, base_mask, 10.0),
            ("jp_data+JODC", comb_ship, comb_mask, 10.0)):
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
        rmse = np.sqrt(np.mean((pred[m] - vfa[m]) ** 2))
        bias = np.mean(pred[m] - vfa[m])
        print(f"{tag}: n={m.sum():,} RMSE={rmse:.3f} mGal bias={bias:+.3f}")


if __name__ == "__main__":
    main()
