import sys; sys.path.insert(0,"src")
import numpy as np
from three_method_compare import load_region_data
from kf_novel import BathyPriorKF
LON=(141,149); LAT=(32,40)
d=load_region_data(LON,LAT,val_cruise="MR22-04_gravity_dmo",upscale=4)
lon,lat=d["lon_hr"],d["lat_hr"]; p=np.asarray(d["val_points"],float)
base=dict(gravity_lr=d["gravity_lr"].astype(float),ship_15s=d["ship_grav"],train_mask=d["train_mask"],
          lon_hr=lon,lat_hr=lat,lon_range=LON,lat_range=LAT,upscale=4,bathy_15s=d["bathy_15s"],n_iter=6,smooth_sigma=4.0)
sh=(d["ssv_hr"].shape,d["gravity_lr"].shape)
def rmse(m):
    ix=np.clip(np.searchsorted(lon,p[:,0]),0,len(lon)-1); iy=np.clip(np.searchsorted(lat,p[:,1]),0,len(lat)-1)
    dd=m[iy,ix]-p[:,2]; return float(np.sqrt(np.mean(dd*dd)))
print("SSV",rmse(d["ssv_hr"]))
for a in (0.0,0.5):
    k=BathyPriorKF(*sh,(8.,8.),ship_noise_std=10.,alpha=a)
    print("alpha",a,rmse(k.build(**base)))
