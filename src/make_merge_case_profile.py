"""合并页：地形+航迹、剖面、误差、散点。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 7})
d = np.load(OUT / "validation_G_fields.npz")
lon, lat = d["lon"], d["lat"]
val = np.asarray(d["val"], float)[:600]
bathy = np.load(ROOT / "output_real/three_parts/ship_regions/G_Japan.npz")["bathy"]
ix = np.clip(np.searchsorted(lon, val[:,0]), 0, len(lon)-1)
iy = np.clip(np.searchsorted(lat, val[:,1]), 0, len(lat)-1)
fields = {"双线性": d["ssv"][iy,ix], "频域融合": d["freq"][iy,ix],
          "LSC": d["lsc"][iy,ix], "相干保持 KF": d["ckf"][iy,ix]}
styles = {"双线性": ("#7F8C8D","--"), "频域融合": ("#1F77B4","-."),
          "LSC": ("#2CA02C",":"), "相干保持 KF": ("#C0392B","-")}
dist = np.r_[0, np.cumsum(np.hypot(
    np.diff(val[:,0])*111*np.cos(np.radians(val[:,1].mean())),
    np.diff(val[:,1])*111))]
x0 = val[:,0].min()-0.25; x1 = val[:,0].max()+0.25
y0 = val[:,1].min()-0.15; y1 = val[:,1].max()+0.15
sx = (lon>=x0)&(lon<=x1); sy=(lat>=y0)&(lat<=y1)
ext = [lon[sx][0], lon[sx][-1], lat[sy][0], lat[sy][-1]]

fig = plt.figure(figsize=(13.4, 5.4), dpi=220)
# terrain
ax = fig.add_axes([0.04, 0.55, 0.42, 0.39])
ls = LightSource(azdeg=315, altdeg=45)
ax.imshow(ls.shade(bathy[np.ix_(sy,sx)], cmap=plt.cm.Greys, vert_exag=0.5, blend_mode="overlay"),
          extent=ext, origin="lower", aspect="auto")
ax.plot(val[:,0], val[:,1], color="#C0392B", lw=1.1)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("(a) 地形 + 船测航迹", fontsize=8)
# profile
ax = fig.add_axes([0.54, 0.55, 0.42, 0.39])
ax.plot(dist, val[:,2], color="k", lw=1.2, label="船测")
for name, z in fields.items():
    c, ls2 = styles[name]
    ax.plot(dist, z, color=c, lw=1.0, ls=ls2, label=name)
ax.set_xlabel("沿测线距离 (km)", fontsize=7)
ax.set_ylabel("重力异常 (mGal)", fontsize=7)
ax.legend(fontsize=5.5, frameon=False, ncol=2)
ax.set_title("(b) 同航次剖面", fontsize=8)
ax.tick_params(labelsize=6); ax.grid(alpha=0.2)
# error
ax = fig.add_axes([0.04, 0.06, 0.42, 0.35])
for name, z in fields.items():
    c, ls2 = styles[name]
    ax.plot(dist, z - val[:,2], color=c, lw=1.0, ls=ls2, label=name)
ax.axhline(0, color="k", lw=0.7)
ax.set_xlabel("沿测线距离 (km)", fontsize=7); ax.set_ylabel("误差 (mGal)", fontsize=7)
ax.set_title("(c) 剖面误差", fontsize=8)
ax.tick_params(labelsize=6); ax.grid(alpha=0.2)
ax.legend(fontsize=5.2, frameon=False, ncol=2, loc="upper right")
# scatter: all validation points
ax = fig.add_axes([0.54, 0.06, 0.42, 0.35])
vall = np.asarray(d["val"], float)
ixv = np.clip(np.searchsorted(lon, vall[:,0]), 0, len(lon)-1)
iyv = np.clip(np.searchsorted(lat, vall[:,1]), 0, len(lat)-1)
hb = ax.hexbin(vall[:,2], d["ckf"][iyv, ixv], gridsize=45,
               cmap="magma", mincnt=1, bins="log")
lo, hi = np.percentile(vall[:,2], [1,99])
ax.plot([lo,hi],[lo,hi],"--",color="cyan",lw=0.8)
ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
ax.set_xlabel("船测 (mGal)", fontsize=7); ax.set_ylabel("相干保持 KF (mGal)", fontsize=7)
ax.set_title("(d) 船测—模型散点", fontsize=8)
ax.tick_params(labelsize=6)
cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.01)
cb.set_label("log10(点数)", fontsize=6); cb.ax.tick_params(labelsize=5)
fig.savefig(OUT / "fig_merge_case_profile.png", bbox_inches="tight")
print("saved")
