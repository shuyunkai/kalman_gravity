"""日本区局部：地形 + 四算法5–20 km + CKF−LSC差值。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import sys
sys.path.insert(0, str(Path(__file__).parent))
from scale_decomposition import bandpass

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 8})
d = np.load(OUT / "validation_G_fields.npz")
lon, lat = d["lon"], d["lat"]
bathy = np.load(ROOT / "output_real/three_parts/ship_regions/G_Japan.npz")["bathy"]
dx = (lon[1]-lon[0])*111320*np.cos(np.radians(lat.mean()))
dy = (lat[1]-lat[0])*111320
sx = (lon >= 139.2) & (lon <= 140.4); sy = (lat >= 34.5) & (lat <= 35.6)
ext = [lon[sx][0], lon[sx][-1], lat[sy][0], lat[sy][-1]]
flds = [("双线性", bandpass(d["ssv"], dx, dy, 5e3, 20e3)),
        ("频域融合", bandpass(d["freq"], dx, dy, 5e3, 20e3)),
        ("LSC", bandpass(d["lsc"], dx, dy, 5e3, 20e3)),
        ("相干保持 KF", bandpass(d["ckf"], dx, dy, 5e3, 20e3))]
fig, axes = plt.subplots(2, 3, figsize=(13.4, 5.0), dpi=200,
                         gridspec_kw={"wspace": 0.05, "hspace": 0.12})
ax = axes[0, 0]
ls = LightSource(azdeg=315, altdeg=45)
z = bathy[np.ix_(sy, sx)]
ax.imshow(ls.shade(z, cmap=plt.cm.Greys, vert_exag=0.5, blend_mode="overlay"),
          extent=ext, origin="lower", aspect="auto")
ax.set_title("(a) 地形", fontsize=8)
axes_list = [axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]
v = np.nanpercentile(np.abs(np.concatenate([f[np.ix_(sy, sx)].ravel() for _, f in flds])), 98)
for ax, (name, f) in zip(axes_list, flds):
    zz = f[np.ix_(sy, sx)]
    ax.imshow(zz, cmap="RdBu_r", extent=ext, origin="lower",
              vmin=-v, vmax=v, aspect="auto")
    ax.set_title(name + "：5–20 km", fontsize=8)
ax = axes[1, 2]
diff = flds[3][1] - flds[2][1]
zz = diff[np.ix_(sy, sx)]
vd = np.nanpercentile(np.abs(zz), 98)
im = ax.imshow(zz, cmap="PuOr_r", extent=ext, origin="lower",
               vmin=-vd, vmax=vd, aspect="auto")
ax.set_title("(f) 相干保持 KF − LSC", fontsize=8)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.01).ax.tick_params(labelsize=5)
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
fig.savefig(OUT / "fig_new_japan_methods_zoom.png", bbox_inches="tight")
print("saved")
