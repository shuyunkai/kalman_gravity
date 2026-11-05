"""第4页：通栏三图（地形/1′/15″），无表格拼图。"""
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
                     "axes.unicode_minus": False, "font.size": 7})
d = np.load(OUT / "validation_G_fields.npz")
lon, lat = d["lon"], d["lat"]
bathy = np.load(ROOT / "output_real/three_parts/ship_regions/G_Japan.npz")["bathy"]
dx = (lon[1]-lon[0])*111320*np.cos(np.radians(lat.mean()))
dy = (lat[1]-lat[0])*111320
ext = [138, 142, 33.5, 36]
sx = (lon >= ext[0]) & (lon <= ext[1]); sy = (lat >= ext[2]) & (lat <= ext[3])
g1 = bandpass(gaussian_filter(d["ssv"], 8e3/dx), dx, dy, 5e3, 20e3)[np.ix_(sy, sx)]
g2 = bandpass(d["ckf"], dx, dy, 5e3, 20e3)[np.ix_(sy, sx)]
fig, axes = plt.subplots(1, 3, figsize=(13.4, 2.2), dpi=240,
                         gridspec_kw={"wspace": 0.06})
ls = LightSource(azdeg=315, altdeg=45)
axes[0].imshow(ls.shade(bathy[np.ix_(sy, sx)], cmap=plt.cm.Greys,
                        vert_exag=0.5, blend_mode="overlay"),
               extent=ext, origin="lower", aspect="auto")
v1 = np.nanpercentile(np.abs(g1), 98)
v2 = np.nanpercentile(np.abs(g2), 98)
im1 = axes[1].imshow(g1, cmap="RdBu_r", extent=ext, origin="lower",
                     vmin=-v1, vmax=v1, aspect="auto")
im2 = axes[2].imshow(g2, cmap="RdBu_r", extent=ext, origin="lower",
                     vmin=-v2, vmax=v2, aspect="auto")
for ax, lab in zip(axes, ["(a)", "(b)", "(c)"]):
    ax.set_xticks([138, 140, 142]); ax.set_yticks([33.5, 34.75, 36])
    ax.tick_params(labelsize=5.5, pad=1)
    for s in ax.spines.values(): s.set_linewidth(0.5)
    ax.text(0.01, 1.01, lab, transform=ax.transAxes, va="bottom", fontsize=7)
cax1 = fig.add_axes([0.36, 0.02, 0.22, 0.025])
cb1 = fig.colorbar(im1, cax=cax1, orientation="horizontal")
cb1.set_label("1′ 5–20 km (mGal)", fontsize=5.5)
cb1.ax.tick_params(labelsize=4.5)
cax2 = fig.add_axes([0.66, 0.02, 0.22, 0.025])
cb2 = fig.colorbar(im2, cax=cax2, orientation="horizontal")
cb2.set_label("15″ 5–20 km (mGal)", fontsize=5.5)
cb2.ax.tick_params(labelsize=4.5)
fig.savefig(OUT / "fig_limitation_wide.png", bbox_inches="tight")
print("saved")
