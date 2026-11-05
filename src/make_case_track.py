"""单航次案例：地形 + 航迹 + 15″ 场上的船测误差。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 8})

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
m = (val[:, 0] >= x0) & (val[:, 0] <= x1) & (val[:, 1] >= y0) & (val[:, 1] <= y1)

fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.8), dpi=220,
                         gridspec_kw={"wspace": 0.08})
ax = axes[0]
ls = LightSource(azdeg=315, altdeg=45)
z = bathy[np.ix_(sy, sx)]
ax.imshow(ls.shade(z, cmap=plt.cm.Greys, vert_exag=0.5, blend_mode="overlay"),
          extent=ext, origin="lower", aspect="auto")
ax.plot(val[m, 0], val[m, 1], color="#C0392B", lw=1.0)
ax.scatter(val[m, 0], val[m, 1], s=2, color="#C0392B")
ax.set_title("(a) 地形与船测航迹", fontsize=9)
ax.set_xticks([]); ax.set_yticks([])

ax = axes[1]
v = np.nanpercentile(np.abs(ckf[np.ix_(sy, sx)]), 98)
ax.imshow(ckf[np.ix_(sy, sx)], cmap="RdBu_r", extent=ext, origin="lower",
          vmin=-v, vmax=v, alpha=0.65, aspect="auto")
sc = ax.scatter(val[m, 0], val[m, 1], c=err[m], cmap="RdYlGn_r",
                vmin=0, vmax=8, s=9, edgecolors="none")
ax.set_title("(b) 15″ 场与船测误差", fontsize=9)
ax.set_xticks([]); ax.set_yticks([])
cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
cb.set_label("|模型−船测| (mGal)", fontsize=7); cb.ax.tick_params(labelsize=6)
fig.savefig(OUT / "fig_new_case_track.png", bbox_inches="tight")
print("saved")
