import sys, numpy as np, xarray as xr
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0,'src')
from three_method_compare import load_region_data
from aniso_kf_test import build as build_aniso
REG={'D_Nankai':((124.,132.),(24.,31.),'YK14-14_gravity_dmo'),'H_Kuroshio':((134.,141.),(31.,36.),'YK18-14_gravity_dmo')}
def rmse(g,lon,lat,p):
 ix=np.clip(np.searchsorted(lon,p[:,0]),0,len(lon)-1); iy=np.clip(np.searchsorted(lat,p[:,1]),0,len(lat)-1)
 return float(np.sqrt(np.mean((g[iy,ix]-p[:,2])**2)))
for name,(lr,la,val) in REG.items():
 d=load_region_data(lr,la,val_cruise=val,upscale=4); lon,lat=d['lon_hr'],d['lat_hr']; pts=np.asarray(d['val_points'],float)
 ds=xr.open_dataset('data/SDUST2022/SDUST2022GRA.nc'); sub=ds['ga'].sel(lon=slice(lr[0],lr[1]),lat=slice(la[0],la[1])).values
 glon=np.asarray(ds.lon.sel(lon=slice(lr[0],lr[1]))); glat=np.asarray(ds.lat.sel(lat=slice(la[0],la[1]))); ds.close()
 if glat[0]>glat[-1]: sub=sub[::-1]; glat=glat[::-1]
 f=RegularGridInterpolator((glat,glon),sub,bounds_error=False,fill_value=np.nan); LLON,LLAT=np.meshgrid(lon,lat)
 pr=f(np.column_stack([LLAT.ravel(),LLON.ravel()])).reshape(LLAT.shape); pr=np.where(np.isfinite(pr),pr,np.nanmedian(pr))
 for p0 in [3.,5.]:
  g=build_aniso(d['gravity_lr'].astype(float),d['bathy_15s'],lon,lat,w_keep=1.,p0=2.,n_iter=6,ship=d['ship_grav'],train=d['train_mask'],ssv_mode='obs_only',p0_obs=p0,region_size_deg=float(lat[-1]-lat[0]),prior_long=pr)
  print(name,'p0',p0,'RMSE',rmse(g,lon,lat,pts),flush=True)
