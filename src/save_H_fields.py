"""保存九州东侧（H）四种 15″ 场，用于 CDF/剖面。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from three_method_compare import load_region_data
from aniso_kf_test import build
from freq_fusion_baseline import freq_fusion_reconstruct
from regions_accuracy_extra import lsc_field

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
LON, LAT = (124.0, 132.0), (24.0, 31.0)

d = load_region_data(LON, LAT, val_cruise="YK14-14_gravity_dmo", upscale=4)
lon, lat = d["lon_hr"], d["lat_hr"]
dx_km = (lon[1] - lon[0]) * 111.0 * np.cos(np.radians(lat.mean()))
freq = freq_fusion_reconstruct(d["ssv_hr"], d["ship_grav"], d["train_mask"],
                               lon, lat, LON, LAT)
lsc = lsc_field(d["ssv_hr"], d["ship_grav"], d["train_mask"], dx_km)
ckf = build(d["gravity_lr"].astype(float), d["bathy_15s"], lon, lat,
            w_keep=1.0, p0=2.0, n_iter=6, ship=d["ship_grav"],
            train=d["train_mask"])
np.savez(OUT / "validation_D_fields.npz", lon=lon, lat=lat,
         ssv=d["ssv_hr"], freq=freq, lsc=lsc, ckf=ckf,
         val=np.asarray(d["val_points"], float))
print("saved", ckf.shape)
