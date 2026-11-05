"""数据基础页：SSV 1′、船测、GEBCO、1′→15″网格。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize
from matplotlib.cm import ScalarMappable

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 7})
d = np.load(OUT / "validation_G_fields.npz")
lon, lat, ssv, val = d["lon"], d["lat"], d["ssv"], d["val"]
bathy = np.load(ROOT / "output_real/three_parts/ship_regions/G_Japan.npz")["bathy"]

# 1' background by 4x4 block average
z = ssv[:ssv.shape[0]//4*4, :ssv.shape[1]//4*4]
blk = z.reshape(z.shape[0]//4, 4, z.shape[1]//4, 4).mean(axis=(1, 3))
blon = lon[::4][:blk.shape[1]]; blat = lat[::4][:blk.shape[0]]
step = 5
fig = plt.figure(figsize=(13.4, 5.1), dpi=220)

# (a) SSV 1'
ax = fig.add_axes([0.04, 0.52, 0.40, 0.40])
v = np.nanpercentile(np.abs(blk), 98)
im = ax.imshow(blk[::step, ::step], cmap="RdBu_r", origin="lower",
               extent=[blon[0], blon[-1], blat[0], blat[-1]],
               vmin=-v, vmax=v, aspect="auto")
ax.set_title("(a) SSV33.1：1′ 块观测", fontsize=8)
ax.set_xticks([]); ax.set_yticks([])
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01)
cb.set_label("mGal", fontsize=6); cb.ax.tick_params(labelsize=5)

# (b) ship points over terrain
ax = fig.add_axes([0.55, 0.52, 0.40, 0.40])
lsb = LightSource(azdeg=315, altdeg=45)
bzb = bathy[::5, ::5]
ax.imshow(lsb.shade(bzb, cmap=plt.cm.Greys, vert_exag=0.4, blend_mode="overlay"),
          origin="lower", extent=[lon[0], lon[-1], lat[0], lat[-1]],
          aspect="auto", alpha=0.85)
sc = ax.scatter(val[:,0], val[:,1], c=val[:,2], s=0.9, cmap="RdYlBu_r",
                vmin=np.percentile(val[:,2],2), vmax=np.percentile(val[:,2],98),
                linewidths=0, alpha=0.95)
ax.set_xlim(lon[0], lon[-1]); ax.set_ylim(lat[0], lat[-1])
ax.set_title("(b) 船测重力：叠地形", fontsize=8)
ax.set_xticks([]); ax.set_yticks([])
cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.01)
cb.set_label("mGal", fontsize=6); cb.ax.tick_params(labelsize=5)

# (c) terrain
ax = fig.add_axes([0.04, 0.07, 0.40, 0.36])
ls = LightSource(azdeg=315, altdeg=45)
bz = bathy[::5, ::5]
ax.imshow(ls.shade(bz, cmap=plt.cm.Greys, vert_exag=0.4, blend_mode="overlay"),
          origin="lower", extent=[lon[0], lon[-1], lat[0], lat[-1]], aspect="auto")
ax.set_title("(c) GEBCO：15″ 地形", fontsize=8)
ax.set_xticks([]); ax.set_yticks([])
sm = ScalarMappable(norm=Normalize(vmin=np.nanmin(bz), vmax=np.nanmax(bz)), cmap="Greys")
cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.01)
cb.set_label("水深 (m)", fontsize=6); cb.ax.tick_params(labelsize=5)

# (d) 1' to 15"
ax = fig.add_axes([0.55, 0.07, 0.40, 0.36])
ax.set_xlim(0, 4.5); ax.set_ylim(-0.8, 4.5); ax.set_aspect("equal")
ax.axis("off")
for i in range(5):
    ax.plot([0.2, 4.2], [0.2+i, 0.2+i], color="#90A4AE", lw=0.7)
    ax.plot([0.2+i, 0.2+i], [0.2, 4.2], color="#90A4AE", lw=0.7)
ax.add_patch(plt.Rectangle((0.2, 0.2), 4, 4, fill=False, ec="#C0392B", lw=1.6))
for i in range(4):
    for j in range(4):
        ax.fill_between([0.2+i,0.2+i+1],[0.2+j,0.2+j],[0.2+j+1,0.2+j+1],
                        color="#EEF3F9", alpha=0.7, zorder=0)
ax.annotate("", xy=(4.2, -0.05), xytext=(0.2, -0.05),
            arrowprops=dict(arrowstyle="<->", color="#0B2545", lw=1.0))
ax.text(2.2, -0.50, "1′ 块 = 4×4 个 15″ 格点（15″ ≈ 0.46 km）", ha="center", fontsize=7)
ax.text(2.2, 2.1, "1′", ha="center", va="center", fontsize=18, color="#C0392B", alpha=0.75)
ax.text(2.2, 4.35, "(d) 1′ → 15″ 格网关系", ha="center", fontsize=8)

fig.savefig(OUT / "fig_data_foundation.png", bbox_inches="tight")
print("saved")

