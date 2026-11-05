"""三个 PPT 区域：XGM2019e 独立长波先验 + SSV 只作块观测（B+XGM）。"""
import json, sys, time
from pathlib import Path
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0, str(Path(__file__).parent))
from three_method_compare import load_region_data
from aniso_kf_test import build as build_aniso

ROOT=Path(__file__).parent.parent
OUT=ROOT/'output_real/three_parts/newdeck'; OUT.mkdir(parents=True,exist_ok=True)
SR=ROOT/'output_real/three_parts/ship_regions'
REGIONS={
 'D_Nankai':dict(lon=(124.0,132.0),lat=(24.0,31.0),val='YK14-14_gravity_dmo'),
 'H_Kuroshio':dict(lon=(134.0,141.0),lat=(31.0,36.0),val='YK18-14_gravity_dmo'),
 'G_Japan':dict(lon=(132.0,152.0),lat=(26.0,42.0),val='MR22-04_gravity_dmo'),
 'F_PhilSea':dict(lon=(139.0,152.0),lat=(13.0,26.0),val='YK18-08_gravity_dmo'),
}
def rmse_at(model,lon,lat,pts):
    ix=np.clip(np.searchsorted(lon,pts[:,0]),0,len(lon)-1)
    iy=np.clip(np.searchsorted(lat,pts[:,1]),0,len(lat)-1)
    d=model[iy,ix]-pts[:,2]
    return float(np.sqrt(np.mean(d**2)))
def xgm_on(lon,lat,lon_range,lat_range):
    ds=xr.open_dataset(ROOT/'data/icgem/grid/XGM2019e_gravity_anomaly.nc')
    sub=ds['gravity_anomaly'].sel(lon=slice(lon_range[0],lon_range[1]),
                                  lat=slice(lat_range[1],lat_range[0])).values
    glon=np.asarray(ds.lon.sel(lon=slice(lon_range[0],lon_range[1])))
    glat=np.asarray(ds.lat.sel(lat=slice(lat_range[1],lat_range[0])))
    ds.close()
    if glat[0]>glat[-1]: sub=sub[::-1]; glat=glat[::-1]
    f=RegularGridInterpolator((glat,glon),sub,bounds_error=False,fill_value=np.nan)
    LLON,LLAT=np.meshgrid(lon,lat)
    z=f(np.column_stack([LLAT.ravel(),LLON.ravel()])).reshape(LLAT.shape)
    return np.where(np.isfinite(z),z,np.nanmedian(z))
res={}
for name,cfg in REGIONS.items():
    print('===',name,flush=True)
    d=load_region_data(cfg['lon'],cfg['lat'],val_cruise=cfg['val'],upscale=4)
    lon,lat=d['lon_hr'],d['lat_hr']; pts=np.asarray(d['val_points'],float)
    xgm=xgm_on(lon,lat,cfg['lon'],cfg['lat'])
    t=time.time()
    g=build_aniso(d['gravity_lr'].astype(float),d['bathy_15s'],lon,lat,
                  w_keep=1.0,p0=2.0,n_iter=6,ship=d['ship_grav'],train=d['train_mask'],
                  ssv_mode='obs_only',p0_obs=10.0,region_size_deg=float(lat[-1]-lat[0]),
                  prior_long=xgm)
    e=rmse_at(g,lon,lat,pts); ssv=rmse_at(d['ssv_hr'],lon,lat,pts)
    print(f'  SSV {ssv:.4f}  B+XGM {e:.4f}  n={len(pts)}  {time.time()-t:.1f}s',flush=True)
    res[name]=dict(ssv=ssv,b_xgm=e,n=int(len(pts)),time_s=round(time.time()-t,1))
    if name in ('D_Nankai','G_Japan','F_PhilSea'):
        f=SR/f'{name}.npz'; old=np.load(f)
        np.savez_compressed(f,lon=lon,lat=lat,ssv=old['ssv'],kf=old['kf'],kf_c=old.get('kf_c',old['kf']),
                            b_ckf=old.get('b_ckf',old['kf']),b_xgm=g.astype(np.float32),bathy=old['bathy'])
    else:
        np.savez_compressed(SR/f'{name}.npz',lon=lon,lat=lat,ssv=d['ssv_hr'].astype(np.float32),
                            b_ckf=g.astype(np.float32),bathy=d['bathy_15s'].astype(np.float32),b_xgm=g.astype(np.float32))
(OUT/'regions_xgm_ckf.json').write_text(json.dumps(res,ensure_ascii=False,indent=1),encoding='utf-8')
print(json.dumps(res,ensure_ascii=False,indent=1))
