import sys; sys.path.insert(0,"src")
import numpy as np
from three_method_compare import load_region_data
from kf_novel import BathyPriorKF
LON=(141,149); LAT=(32,40)
def rmse(m,lon,lat,p):
    ix=np.clip(np.searchsorted(lon,p[:,0]),0,len(lon)-1); iy=np.clip(np.searchsorted(lat,p[:,1]),0,len(lat)-1)
    return float(np.sqrt(np.mean((m[iy,ix]-p[:,2])**2)))
for c in ("MR23-02_gravity_dmo","MR21-02_gravity_dmo"):
    d=load_region_data(LON,LAT,val_cruise=c,upscale=4)
    lon,lat=d["lon_hr"],d["lat_hr"]; p=np.asarray(d["val_points"],float)
    base=dict(gravity_lr=d["gravity_lr"].astype(float),ship_15s=d["ship_grav"],train_mask=d["train_mask"],
              lon_hr=lon,lat_hr=lat,lon_range=LON,lat_range=LAT,upscale=4,bathy_15s=d["bathy_15s"],n_iter=6,smooth_sigma=4.0)
    sh=(d["ssv_hr"].shape,d["gravity_lr"].shape)
    k=BathyPriorKF(*sh,(8.,8.),ship_noise_std=10.,alpha=0.5); e0=rmse(k.build(**base),lon,lat,p)
    b=BathyPriorKF(*sh,(8.,8.),ship_noise_std=10.,alpha=0.5,p0_std=4.,q_std=4.,adaptive_ship=False)
    e1=rmse(b.build(**base,localize=True,adaptive=True,corr_px=28.0),lon,lat,p)
    print(f"{c}: KF {e0:.4f} BOL28 {e1:.4f} diff {e1-e0:+.4f}")
