import sys, time, json
import numpy as np
sys.path.insert(0,'src')
from kf_novel import BathyPriorKF, block_fill
from kalman_gravity import KalmanGravityReconstructor
from independent_validation_v2 import evaluate_15s_raw

D=np.load('output_real/three_parts/abc_region_inputs.npz')
lr=D['gravity_lr'].astype(float); ship=D['ship_grav'].astype(float)
train=D['train_mask'].astype(bool); bathy=D['bathy_15s'].astype(float)
lon_hr=D['lon_hr']; lat_hr=D['lat_hr']; val=D['val_points'].astype(float)
ny,nx=1920,1920; n=ny*nx

kf=BathyPriorKF((ny,nx),lr.shape,8.0,process_noise_std=2.0,sat_noise_std=3.0,ship_noise_std=1.0,alpha=0.5,adaptive_ship=False)
ssv_block=block_fill(lr,4)
bathy_hp=kf._high_pass(kf._bathy_to_bouguer(bathy,lon_hr,lat_hr),lon_hr,lat_hr)
priorC=ssv_block+0.5*bathy_hp
priorB=0.5*bathy_hp

def run_case(name,prior,sat=True):
    print('\n===',name,'sat=',sat,'===')
    r=KalmanGravityReconstructor(grid_shape=(ny,nx),region_size_deg=8.0,lr_grid_shape=lr.shape,
        localize=False,adaptive=False,corr_px=3.0,adaptive_ship=False)
    r._initialized=True; r.ny=ny; r.nx=nx; r.n_state=n
    r.x=prior.ravel().copy()
    r.P_diag=(2500.0 if name=='B_obs_only' else 4.0)*np.ones(n); r.Q_diag=4.0*np.ones(n)
    r.R_sat_val=9.0; r.R_ship_val=1.0
    r.gravity_lr=lr; r.gravity_ship=ship; r.ship_train_mask=train; r._iter_count=0; r.ship_inc_history=[]
    if not sat:
        r.update_satellite=lambda: None
    t=time.time(); g=r.run(n_iterations=6,smooth_sigma=4.0,verbose=True); dt=time.time()-t
    m=evaluate_15s_raw(g,val,lon_hr,lat_hr)
    # 1' block mean consistency vs SSV
    bm=g.reshape(480,4,480,4).mean(axis=(1,3))
    m['block_rmse']=float(np.sqrt(np.mean((bm-lr)**2)))
    m['block_bias']=float(np.mean(bm-lr))
    m['time_s']=dt
    print(name,json.dumps({k:round(v,4) if isinstance(v,float) else v for k,v in m.items()},ensure_ascii=False))
    return g,m

res={}
for name,prior,sat in [('A_prior_only',priorC,False),('B_obs_only',priorB,True),('C_both',priorC,True)]:
    g,m=run_case(name,prior,sat)
    res[name]=(g,m)
    np.save(f'output_real/three_parts/abc_{name}.npy',g)

def dstat(a,b):
    d=a-b; return {'rmse':float(np.sqrt(np.mean(d*d))),'bias':float(d.mean()),'maxabs':float(np.max(np.abs(d)))}
summary={k:v[1] for k,v in res.items()}
summary['C_minus_A']=dstat(res['C_both'][0],res['A_prior_only'][0])
summary['C_minus_B']=dstat(res['C_both'][0],res['B_obs_only'][0])
summary['A_minus_B']=dstat(res['A_prior_only'][0],res['B_obs_only'][0])
with open('output_real/three_parts/abc_metrics.json','w',encoding='utf-8') as f: json.dump(summary,f,ensure_ascii=False,indent=2)
print('DONE')


