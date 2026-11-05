import numpy as np, json
from scipy.interpolate import RegularGridInterpolator
D=np.load('output_real/three_parts/abc_region_inputs.npz')
lon=D['lon_hr']; lat=D['lat_hr']; val=D['val_points'].astype(float)
fields={k:np.load(f'output_real/three_parts/abc_{k}.npy') for k in ['A_prior_only','B_obs_only','C_both']}
errs={}
for k,g in fields.items():
    f=RegularGridInterpolator((lat,lon),g,bounds_error=False,fill_value=np.nan)
    pred=f(np.column_stack([val[:,1],val[:,0]])); truth=val[:,2]
    ok=~(np.isnan(pred)|np.isnan(truth)); errs[k]=(pred[ok]-truth[ok]); print(k,'n',ok.sum())
idx=np.arange(len(errs['A_prior_only']))
rng=np.random.default_rng(42); nboot=2000
def rmse(e): return float(np.sqrt(np.mean(e*e)))
out={}
for a,b in [('C_both','B_obs_only'),('C_both','A_prior_only'),('B_obs_only','A_prior_only')]:
    ea,eb=errs[a],errs[b]
    obs=rmse(eb)-rmse(ea)  # positive = a better (lower RMSE)
    bs=np.empty(nboot)
    for i in range(nboot):
        ii=rng.integers(0,len(ea),len(ea)); bs[i]=rmse(eb[ii])-rmse(ea[ii])
    out[f'{a}_vs_{b}']={'delta_rmse':obs,'ci95':[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],'p_boot':float(np.mean(bs<=0))}
print(json.dumps(out,indent=2))
json.dump(out,open('output_real/three_parts/abc_bootstrap.json','w'),indent=2)
