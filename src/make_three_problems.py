import sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0,'src')
from scale_decomposition import bandpass
from matplotlib.colors import LightSource
D=np.load('output_real/three_parts/newdeck/validation_G_fields.npz')
lon,lat,val=D['lon'],D['lat'],np.asarray(D['val'],float)
bathy=np.load('output_real/three_parts/ship_regions/G_Japan.npz')['bathy']
g=D['ckf']; dx=(lon[1]-lon[0])*111320*np.cos(np.radians(lat.mean())); dy=(lat[1]-lat[0])*111320
gb=bandpass(g,dx,dy,5e3,20e3); tb=bandpass(0.0688*bathy,dx,dy,5e3,20e3)
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':7})
fig,axs=plt.subplots(1,3,figsize=(8.4,2.45),dpi=220,constrained_layout=True)
# (a) block diagram
ax=axs[0]; ax.set_xlim(0,4.6); ax.set_ylim(-0.7,4.5); ax.set_aspect('equal'); ax.axis('off')
for i in range(5):
 ax.plot([.2,4.2],[.2+i,.2+i],color='#90A4AE',lw=.8); ax.plot([.2+i,.2+i],[.2,4.2],color='#90A4AE',lw=.8)
ax.add_patch(Rectangle((.2,.2),4,4,fill=False,ec='#C0392B',lw=1.6))
for i in range(4):
 for j in range(4):
  ax.fill_between([.2+i,.2+i+1],[.2+j,.2+j],[.2+j+1,.2+j+1],color='#EEF3F9',alpha=.8,zorder=0)
ax.text(2.2,2.1,'1′',ha='center',va='center',fontsize=16,color='#C0392B',alpha=.8)
ax.text(2.2,-.35,'1′ 块 = 16 个 15″格点，只有 1 个块均值',ha='center',fontsize=6.5)
ax.set_title('(a) 1′块内无直接观测',fontsize=8)
# (b) ship tracks
ax=axs[1]; ls=LightSource(azdeg=315,altdeg=45); bz=bathy[::15,::15]
ax.imshow(ls.shade(bz,cmap=plt.cm.Greys,vert_exag=.4,blend_mode='overlay'),origin='lower',extent=[lon[0],lon[-1],lat[0],lat[-1]],aspect='auto',alpha=.7)
ax.scatter(val[::10,0],val[::10,1],s=.3,c='#C0392B',linewidths=0,alpha=.8)
ax.set_xlim(139,145); ax.set_ylim(34,40); ax.set_xticks([]); ax.set_yticks([]); ax.set_title('(b) 船测沿航迹、覆盖不均',fontsize=8)
# (c) hexbin terrain vs gravity
ax=axs[2]; m=np.isfinite(gb)&np.isfinite(tb); x=tb[m][::8]; y=gb[m][::8]
hb=ax.hexbin(x,y,gridsize=35,cmap='viridis',mincnt=1); ax.set_xlabel('地形高频 (mGal)',fontsize=6); ax.set_ylabel('5–20 km 重力 (mGal)',fontsize=6)
r=np.corrcoef(x,y)[0,1]; ax.set_title(f'(c) 地形—重力相关但非一一对应  r={r:.2f}',fontsize=7.3); ax.tick_params(labelsize=5)
fig.savefig('output_real/three_parts/newdeck/fig_three_problems.png',bbox_inches='tight')
print('saved',r)
