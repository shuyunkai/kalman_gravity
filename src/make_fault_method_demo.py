import sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,'src')
from scale_decomposition import bandpass, thg, lineament_mask, filter_lineaments
ROOT='output_real/three_parts/newdeck/validation_G_fields.npz'
d=np.load(ROOT); lon=d['lon']; lat=d['lat']; g=d['ckf']
dx=(lon[1]-lon[0])*111320*np.cos(np.radians(lat.mean())); dy=(lat[1]-lat[0])*111320
b=bandpass(g,dx,dy,5e3,20e3)
t=thg(b,dx,dy)*1000
m=filter_lineaments(lineament_mask(b,dx,dy,pct=60),20,1.8)
sx=(lon>=139.2)&(lon<=140.4); sy=(lat>=34.5)&(lat<=35.6)
ext=[lon[sx][0],lon[sx][-1],lat[sy][0],lat[sy][-1]]
zz=[b[np.ix_(sy,sx)],t[np.ix_(sy,sx)]]
mask=m[np.ix_(sy,sx)]
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':8})
fig,axs=plt.subplots(1,3,figsize=(8.0,2.35),dpi=220,constrained_layout=True)
for ax,z,title,cmap in [(axs[0],zz[0],'(a) 输入场（5–20 km）','RdBu_r'),(axs[1],zz[1],'(b) THG 水平总梯度','magma'),(axs[2],zz[1],'(c) THG 高值脊线','magma')]:
    v=np.nanpercentile(np.abs(z),98); im=ax.imshow(z,origin='lower',extent=ext,aspect='auto',cmap=cmap,vmin=(-v if cmap=='RdBu_r' else np.nanmin(z)),vmax=(v if cmap=='RdBu_r' else np.nanmax(z)))
    ax.set_title(title,fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im,ax=ax,fraction=0.04,pad=0.01).ax.tick_params(labelsize=5)
ys,xs=np.where(mask)
ax=axs[2]; ax.scatter(lon[sx][xs],lat[sy][ys],s=.25,c='cyan',linewidths=0,alpha=.8)
ax.scatter([],[],s=8,c='cyan',label='线性体像元'); ax.legend(fontsize=5.5,frameon=False,loc='lower right')
fig.savefig('output_real/three_parts/newdeck/fig_fault_method_demo.png',bbox_inches='tight')
print('saved')

