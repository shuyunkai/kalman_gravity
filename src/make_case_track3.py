"""单航次：地形、15″重力场、船测误差三图。"""
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
ckf = d["ckf"]
val = np.asarray(d["val"], float)[:600]
bathy = np.load(ROOT / "output_real/three_parts/ship_regions/G_Japan.npz")["bathy"]
ix = np.clip(np.searchsorted(lon, val[:, 0]), 0, len(lon)-1)
iy = np.clip(np.searchsorted(lat, val[:, 1]), 0, len(lat)-1)
err = np.abs(ckf[iy, ix] - val[:, 2])
x0 = val[:,0].min()-0.25; x1 = val[:,0].max()+0.25
y0 = val[:,1].min()-0.15; y1 = val[:,1].max()+0.15
sx = (lon >= x0) & (lon <= x1); sy = (lat >= y0) & (lat <= y1)
ext = [lon[sx][0], lon[sx][-1], lat[sy][0], lat[sy][-1]]

fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.6), dpi=220,
                         gridspec_kw={"wspace": 0.08})
# terrain
ls = LightSource(azdeg=315, altdeg=45)
z = bathy[np.ix_(sy, sx)]
axes[0].imshow(ls.shade(z, cmap=plt.cm.Greys, vert_exag=0.5, blend_mode="overlay"),
               extent=ext, origin="lower", aspect="auto")
axes[0].plot(val[:,0], val[:,1], color="#C0392B", lw=1.2)
axes[0].set_title("(a) 地形 + 航迹", fontsize=8)
# gravity field
v = np.nanpercentile(np.abs(ckf[np.ix_(sy, sx)]), 98)
im = axes[1].imshow(ckf[np.ix_(sy, sx)], cmap="RdBu_r", extent=ext,
                    origin="lower", vmin=-v, vmax=v, aspect="auto")
axes[1].plot(val[:,0], val[:,1], color="k", lw=0.9)
axes[1].set_title("(b) 15″ 重力异常 + 航迹", fontsize=8)
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.01).set_label("mGal", fontsize=6)
# error
axes[2].imshow(ckf[np.ix_(sy, sx)], cmap="Greys", extent=ext, origin="lower",
               vmin=-v, vmax=v, alpha=0.45, aspect="auto")
sc = axes[2].scatter(val[:,0], val[:,1], c=err, cmap="RdYlGn_r",
                     vmin=0, vmax=8, s=9, edgecolors="none")
axes[2].set_title("(c) 15″ 场 + 船测误差", fontsize=8)
fig.colorbar(sc, ax=axes[2], fraction=0.046, pad=0.01).set_label("|模型−船测| (mGal)", fontsize=6)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.savefig(OUT / "fig_new_case_track3.png", bbox_inches="tight")
print("saved")
