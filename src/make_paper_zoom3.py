"""论文式局部放大：地形、1′ 重力、15″ 重力。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
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
bathy = np.load(ROOT / "output_real/three_parts/ship_regions/G_Japan.npz")["bathy"]
sx = (lon >= 138) & (lon <= 142); sy = (lat >= 33.5) & (lat <= 36)
ext = [lon[sx][0], lon[sx][-1], lat[sy][0], lat[sy][-1]]

fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2), dpi=220,
                         gridspec_kw={"wspace": 0.10})
# terrain hillshade
ls = LightSource(azdeg=315, altdeg=45)
z = bathy[np.ix_(sy, sx)]
rgb = ls.shade(z, cmap=plt.cm.Greys, vert_exag=0.5,
               blend_mode="overlay")
axes[0].imshow(rgb, extent=ext, origin="lower", aspect="auto")
axes[0].set_title("(a) 海底地形", fontsize=9)
# gravity: 5-20 km band
dx = (lon[1]-lon[0])*111320*np.cos(np.radians(lat.mean()))
dy = (lat[1]-lat[0])*111320
g1 = bandpass(gaussian_filter(ssv, 8e3/dx), dx, dy, 5e3, 20e3)
g2 = bandpass(ckf, dx, dy, 5e3, 20e3)
for ax, fld, title in [(axes[1], g1, "(b) 1′ 5–20 km (mGal)"),
                       (axes[2], g2, "(c) 15″ 5–20 km (mGal)")]:
    zz = fld[np.ix_(sy, sx)]
    v = np.nanpercentile(np.abs(zz), 98)
    im = ax.imshow(zz, extent=ext, origin="lower", cmap="RdBu_r",
                   vmin=-v, vmax=v, aspect="auto")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01).ax.tick_params(labelsize=5)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
    ax.add_patch(plt.Rectangle((139.2, 34.6), 1.2, 0.9, fill=False,
                               ec="#C0392B", lw=1.2))
fig.savefig(OUT / "fig_new_paper_zoom3.png", bbox_inches="tight")
print("saved")
