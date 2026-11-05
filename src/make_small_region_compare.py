"""小区域算法对比：四方法 + 差值。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 7})
d = np.load(OUT / "validation_G_fields.npz")
lon, lat = d["lon"], d["lat"]
v = d["val"][:600]
x0, x1 = v[:,0].min()-.25, v[:,0].max()+.25
y0, y1 = v[:,1].min()-.15, v[:,1].max()+.15
sx = (lon >= x0) & (lon <= x1); sy = (lat >= y0) & (lat <= y1)
ext = [lon[sx][0], lon[sx][-1], lat[sy][0], lat[sy][-1]]
flds = [("双线性", "ssv"), ("频域融合", "freq"), ("LSC", "lsc"), ("相干保持 KF", "ckf")]
fig = plt.figure(figsize=(10.5, 5.2), dpi=220)
pos = [[0.05,0.54,0.27,0.36],[0.36,0.54,0.27,0.36],[0.05,0.09,0.27,0.36],[0.36,0.09,0.27,0.36]]
vmax = np.nanpercentile(np.abs(d["ckf"][np.ix_(sy,sx)]),98)
for (name,key),p in zip(flds,pos):
 ax=fig.add_axes(p); z=d[key][np.ix_(sy,sx)]
 im=ax.imshow(z,cmap="RdBu_r",extent=ext,origin="lower",vmin=-vmax,vmax=vmax,aspect="auto")
 ax.set_title(name,fontsize=8); ax.set_xticks([]); ax.set_yticks([])
 if key=="ckf":
  cax=fig.add_axes([0.66,0.09,0.012,0.36]); cb=fig.colorbar(im,cax=cax); cb.set_label("mGal",fontsize=6)
# difference panels
for (name,key),p in zip([("lsc","lsc"),("freq","freq")],[ [0.66,0.54,0.27,0.36],[0.66,0.09,0.27,0.36] ]):
 ax=fig.add_axes(p); z=(d["ckf"]-d[key])[np.ix_(sy,sx)]
 vd=np.nanpercentile(np.abs(z),95)
 im=ax.imshow(z,cmap="PuOr_r",extent=ext,origin="lower",vmin=-vd,vmax=vd,aspect="auto")
 ax.set_title(("相干保持 KF − LSC" if key=="lsc" else "相干保持 KF − 频域"),fontsize=7)
 ax.set_xticks([]); ax.set_yticks([])
fig.savefig(OUT/"fig_small_region_methods.png",bbox_inches="tight")
print("saved")
