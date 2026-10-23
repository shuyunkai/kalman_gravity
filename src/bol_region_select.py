"""每条航次内部交叉验证选相关长度 l，再在留出航次上评估。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from three_method_compare import load_region_data
from kf_novel import BathyPriorKF

LON = (141.0, 149.0)
LAT = (32.0, 40.0)
CAND = (2.0, 4.0, 6.0, 8.0)


def rmse_at(model, lon, lat, pts):
    ix = np.clip(np.searchsorted(lon, pts[:, 0]), 0, len(lon) - 1)
    iy = np.clip(np.searchsorted(lat, pts[:, 1]), 0, len(lat) - 1)
    d = model[iy, ix] - pts[:, 2]
    return float(np.sqrt(np.mean(d ** 2)))


def run(cruise):
    d = load_region_data(LON, LAT, val_cruise=cruise, upscale=4)
    lon, lat = d["lon_hr"], d["lat_hr"]
    ship = d["ship_grav"].copy()
    mask = d["train_mask"].copy()
    val = np.asarray(d["val_points"], float)
    idx = np.where(mask.ravel())[0]
    rng = np.random.default_rng(0)
    perm = rng.permutation(idx)
    hold = perm[:len(perm) // 2]
    keep = perm[len(perm) // 2:]

    ship_in = ship.copy(); ship_in.ravel()[hold] = np.nan
    mask_in = ~np.isnan(ship_in)
    iy, ix = np.divmod(hold, ship.shape[1])
    inner_pts = np.c_[lon[ix], lat[iy], ship.ravel()[hold]]

    def base(g_ship, g_mask):
        return dict(gravity_lr=d["gravity_lr"].astype(float),
                    ship_15s=g_ship, train_mask=g_mask,
                    lon_hr=lon, lat_hr=lat, lon_range=LON, lat_range=LAT,
                    upscale=4, bathy_15s=d["bathy_15s"], n_iter=6,
                    smooth_sigma=4.0)

    shape = (d["ssv_hr"].shape, d["gravity_lr"].shape)
    best = None
    for cp in CAND:
        m = BathyPriorKF(*shape, (8.0, 8.0), ship_noise_std=10.0, alpha=0.5,
                         p0_std=4.0, q_std=4.0, adaptive_ship=False)
        g = m.build(**base(ship_in, mask_in), localize=True, adaptive=True,
                    corr_px=cp)
        e = rmse_at(g, lon, lat, inner_pts)
        print(f"  {cruise} inner cp={cp}: {e:.4f}")
        if best is None or e < best[0]:
            best = (e, cp)
    print(f"  {cruise} best cp={best[1]} (inner {best[0]:.4f})")

    kf = BathyPriorKF(*shape, (8.0, 8.0), ship_noise_std=10.0, alpha=0.5)
    e0 = rmse_at(kf.build(**base(ship, mask)), lon, lat, val)
    bol = BathyPriorKF(*shape, (8.0, 8.0), ship_noise_std=10.0, alpha=0.5,
                       p0_std=4.0, q_std=4.0, adaptive_ship=False)
    e1 = rmse_at(bol.build(**base(ship, mask), localize=True, adaptive=True,
                           corr_px=best[1]), lon, lat, val)
    print(f"{cruise}: KF {e0:.4f}  BOL(cp={best[1]}) {e1:.4f}  "
          f"diff {e1-e0:+.4f}")


if __name__ == "__main__":
    for c in ("MR22-04_gravity_dmo", "MR23-02_gravity_dmo",
              "MR21-02_gravity_dmo"):
        run(c)
