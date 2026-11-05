"""逐轮/RTS 的船测留出 RMSE，用于纵向精度页。"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import aniso_kf_test as ak
from aniso_kf_test import build
from three_method_compare import load_region_data

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
LON, LAT = (132.0, 152.0), (26.0, 42.0)
ak.LON, ak.LAT = LON, LAT

d = load_region_data(LON, LAT, val_cruise="MR22-04_gravity_dmo", upscale=4)
lon, lat = d["lon_hr"], d["lat_hr"]
pts = np.asarray(d["val_points"], float)
r = build(d["gravity_lr"].astype(float), d["bathy_15s"], lon, lat,
          w_keep=1.0, p0=2.0, n_iter=6, ship=d["ship_grav"],
          train=d["train_mask"], return_reconstructor=True)
g = build(d["gravity_lr"].astype(float), d["bathy_15s"], lon, lat,
          w_keep=1.0, p0=2.0, n_iter=6, ship=d["ship_grav"],
          train=d["train_mask"])


def rmse(field):
    ix = np.clip(np.searchsorted(lon, pts[:, 0]), 0, len(lon) - 1)
    iy = np.clip(np.searchsorted(lat, pts[:, 1]), 0, len(lat) - 1)
    e = field[iy, ix] - pts[:, 2]
    return float(np.sqrt(np.mean(e ** 2)))


vals = [rmse(r.prior_2d)] + [rmse(h) for h in r.x_history] + [rmse(g)]
print(vals)
np.savez(OUT / "vertical_rmse_G.npz", vals=np.array(vals))
