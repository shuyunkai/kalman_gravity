"""困难页全地图版：块平均、1′/15″ 5–20 km、差值。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).parent))
from scale_decomposition import bandpass
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 8})

d = np.load(OUT / "validation_G_fields.npz")
lon, lat = d["lon"], d["lat"]
ssv, ckf = d["ssv"], d["ckf"]
dx = (lon[1]-lon[0])*111320*np.cos(np.radians(lat.mean()))
dy = (lat[1]-lat[0])*111320
sx = (lon >= 138) & (lon <= 142); sy = (lat >= 33.5) & (lat <= 36)
ext = [lon[sx][0], lon[sx][-1], lat[sy][0], lat[sy][-1]]
g1 = bandpass(gaussian_filter(ssv, 8e3/dx), dx, dy, 5e3, 20e3)
g2 = bandpass(ckf, dx, dy, 5e3, 20e3)
z1 = g1[np.ix_(sy, sx)]; z2 = g2[np.ix_(sy, sx)]
v = np.nanpercentile(np.abs(np.r_[z1.ravel(), z2.ravel()]), 98)

fig, axes = plt.subplots(2, 2, figsize=(13.4, 5.2), dpi=220)
ax = axes[0, 0]
zz = ssv[np.ix_(sy, sx)]
ax.imshow(zz, cmap="RdBu_r", extent=ext, origin="lower", aspect="auto",
          vmin=-100, vmax=100)
for k in range(0, zz.shape[1], 4):
    ax.axvline(ext[0]+k*(ext[1]-ext[0])/zz.shape[1], color="k", lw=0.2, alpha=0.4)
for k in range(0, zz.shape[0], 4):
    ax.axhline(ext[2]+k*(ext[3]-ext[2])/zz.shape[0], color="k", lw=0.2, alpha=0.4)
ax.set_title("(a) 1′ 块平均：块内无观测", fontsize=8)
ax.set_xticks([]); ax.set_yticks([])

for ax, z, title in [(axes[0, 1], z1, "(b) 1′ 产品：5–20 km 弱"),
                     (axes[1, 0], z2, "(c) 15″ 场：5–20 km 强")]:
    im = ax.imshow(z, cmap="RdBu_r", extent=ext, origin="lower", aspect="auto",
                   vmin=-v, vmax=v)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01).ax.tick_params(labelsize=5)

ax = axes[1, 1]
diff = z2 - z1
vd = np.nanpercentile(np.abs(diff), 98)
im = ax.imshow(diff, cmap="RdBu_r", extent=ext, origin="lower", aspect="auto",
               vmin=-vd, vmax=vd)
ax.set_title("(d) 15″ − 1′：短波差异", fontsize=8)
ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01).ax.tick_params(labelsize=5)
fig.savefig(OUT / "fig_new_hard_maps.png", bbox_inches="tight")
print("saved")
