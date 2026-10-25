import sys; sys.path.insert(0,"src")
import numpy as np
from three_method_compare import load_region_data
from aniso_kf_test import build
LON=(141,149); LAT=(32,40)
d=load_region_data(LON,LAT,val_cruise="MR22-04_gravity_dmo",upscale=4)
lon,lat=d["lon_hr"],d["lat_hr"]; p=np.asarray(d["val_points"],float)
def rmse(m):
    ix=np.clip(np.searchsorted(lon,p[:,0]),0,len(lon)-1); iy=np.clip(np.searchsorted(lat,p[:,1]),0,len(lat)-1)
    return float(np.sqrt(np.mean((m[iy,ix]-p[:,2])**2)))
for w in (0.0,0.5,0.8,1.0):
    g=build(d["gravity_lr"].astype(float),d["bathy_15s"],lon,lat,w_keep=w,p0=2.0,
            n_iter=6,ship=d["ship_grav"],train=d["train_mask"])
    print(w,rmse(g))
