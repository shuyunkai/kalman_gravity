import numpy as np, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
D=np.load('output_real/three_parts/abc_region_inputs.npz')
lon=D['lon_hr']; lat=D['lat_hr']; ext=[lon[0],lon[-1],lat[0],lat[-1]]
G={k:np.load(f'output_real/three_parts/abc_{k}.npy') for k in ['A_prior_only','B_obs_only','C_both']}
M=json.load(open('output_real/three_parts/abc_metrics.json'))
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':9})
fig,axs=plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
titles={'A_prior_only':'(a) A：SSV 只作先验','B_obs_only':'(b) B：SSV 只作块观测','C_both':'(c) C：先验+块观测'}
for ax,k in zip(axs.ravel()[:3],G):
 im=ax.imshow(G[k],origin='lower',extent=ext,cmap='RdBu_r',vmin=-200,vmax=200,aspect='auto')
 ax.set_title(titles[k]); ax.set_xlabel('经度 (°E)'); ax.set_ylabel('纬度 (°N)')
plt.colorbar(im,ax=axs[0,:],shrink=.8,label='重力异常 (mGal)')
for ax,k,t in zip(axs.ravel()[3:5],['C_minus_A','C_minus_B'],['(d) C − A','(e) C − B']):
 d=G['C_both']-G['A_prior_only'] if k=='C_minus_A' else G['C_both']-G['B_obs_only']
 im=ax.imshow(d,origin='lower',extent=ext,cmap='RdBu_r',norm=TwoSlopeNorm(0,-15,15),aspect='auto')
 ax.set_title(t); ax.set_xlabel('经度 (°E)'); ax.set_ylabel('纬度 (°N)')
plt.colorbar(im,ax=axs[1,:2],shrink=.8,label='差值 (mGal)')
ax=axs[1,2]
ks=['A_prior_only','B_obs_only','C_both']; labels=['A','B','C']
x=np.arange(3); w=.36
ax.bar(x-w/2,[M[k]['RMSE'] for k in ks],w,label='验证 RMSE')
ax.bar(x+w/2,[M[k]['block_rmse'] for k in ks],w,label='1′块均值 RMSE')
for i,k in enumerate(ks):
 ax.text(i-w/2,M[k]['RMSE']+.05,f"{M[k]['RMSE']:.2f}",ha='center',fontsize=8)
 ax.text(i+w/2,M[k]['block_rmse']+.05,f"{M[k]['block_rmse']:.2f}",ha='center',fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('mGal'); ax.set_title('(f) 误差对比'); ax.legend(fontsize=8)
fig.savefig('output_real/three_parts/abc_compare.png',dpi=180,bbox_inches='tight')
print('saved')
