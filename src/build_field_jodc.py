"""用 jp_data + JODC（逐航次基准校正）重建日本窗口 15″ 场。"""
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).parent))

from three_method_compare import load_region_data
from kf_novel import BathyPriorKF

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real" / "test_field_15s_jodc.npz"


def main():
    data = load_region_data((141, 149), (32, 40),
                            val_cruise="MR22-04_gravity_dmo", upscale=4)
    ship = data["ship_grav"].astype(float).copy()
    mask = data["train_mask"].copy()
    lon_hr, lat_hr = data["lon_hr"], data["lat_hr"]

    d = np.load(ROOT / "data" / "ship_gravity" / "jodc_fa.npz")
    lon, lat, fa, voy = d["lon"], d["lat"], d["fa"], d["voyage"]
    ssvp = RegularGridInterpolator((lat_hr, lon_hr), data["ssv_hr"],
                                   bounds_error=False, fill_value=np.nan)(
        np.column_stack([lat, lon]))
    fa = fa.copy()
    for v in np.unique(voy):
        sel = (voy == v) & np.isfinite(ssvp)
        if sel.sum() > 50:
            fa[sel] -= np.median(fa[sel] - ssvp[sel])

    ok = np.isfinite(fa)
    ok &= (lon >= lon_hr.min()) & (lon <= lon_hr.max())
    ok &= (lat >= lat_hr.min()) & (lat <= lat_hr.max())
    ix = np.clip(np.searchsorted(lon_hr, lon[ok]), 0, data["nx_hr"] - 1)
    iy = np.clip(np.searchsorted(lat_hr, lat[ok]), 0, data["ny_hr"] - 1)
    free = ~mask[iy, ix]
    ship[iy[free], ix[free]] = fa[ok][free]
    mask[iy, ix] = True
    print("JODC points", ok.sum(), "new cells", int(free.sum()))

    kf = BathyPriorKF(data["ssv_hr"].shape, data["gravity_lr"].shape,
                      (8.0, 8.0), ship_noise_std=10.0, alpha=0.5)
    g = kf.build(data["gravity_lr"].astype(float), ship, mask,
                 lon_hr, lat_hr, (141, 149), (32, 40), upscale=4,
                 bathy_15s=data["bathy_15s"], n_iter=6, smooth_sigma=4.0)
    np.savez_compressed(OUT, g15=g, bathy=data["bathy_15s"],
                        lon=lon_hr, lat=lat_hr)
    print("saved", OUT)


if __name__ == "__main__":
    main()
