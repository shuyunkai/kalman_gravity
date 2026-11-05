import sys, numpy as np, xarray as xr
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0,'src')
from aniso_kf_test import build as build_aniso
from independent_validation_v2 import evaluate_15s_raw
D=np.load('output_real/three_parts/abc_region_inputs.npz')
lr=D['gravity_lr'].astype(float); bathy=D['bathy_15s']; ship=D['ship_grav']
train=D['train_mask']; lon=D['lon_hr']; lat=D['lat_hr']; val=D['val_points'].astype(float)
ds=xr.open_dataset('data/icgem/grid/XGM2019e_gravity_anomaly.nc')
sub=ds['gravity_anomaly'].sel(lon=slice(141,149),lat=slice(40,32)).values
glon=np.asarray(ds.lon.sel(lon=slice(141,149))); glat=np.asarray(ds.lat.sel(lat=slice(40,32)))
ds.close()
if glat[0]>glat[-1]: sub=sub[::-1]; glat=glat[::-1]
f=RegularGridInterpolator((glat,glon),sub,bounds_error=False,fill_value=np.nan)
LLON,LLAT=np.meshgrid(lon,lat)
xgm=f(np.column_stack([LLAT.ravel(),LLON.ravel()])).reshape(LLAT.shape)
xgm=np.where(np.isfinite(xgm),xgm,np.nanmedian(xgm))
print('xgm',xgm.shape,np.nanmin(xgm),np.nanmax(xgm),np.nanstd(xgm))
for p0 in [10.0,20.0,50.0]:
 g=build_aniso(lr,bathy,lon,lat,w_keep=1.0,p0=2.0,n_iter=6,ship=ship,train=train,
               ssv_mode='obs_only',p0_obs=p0,region_size_deg=8.0,prior_long=xgm)
 m=evaluate_15s_raw(g,val,lon,lat)
 print('XGM prior p0',p0,'RMSE',m['RMSE'],'bias',m['Bias'],'STD',m['STD'])
 np.save(f'output_real/three_parts/abc_XGM_p{int(p0)}.npy',g)


