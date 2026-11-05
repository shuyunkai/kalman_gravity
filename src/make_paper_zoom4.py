"""局部放大：地形、1′、15″、15″−1′。"""
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
ssv, ckf = d["ssv"], d["ckf"]
bathy = np.load(ROOT / "output_real/three_parts/ship_regions/G_Japan.npz")["bathy"]
sx = (lon >= 138) & (lon <= 142); sy = (lat >= 33.5) & (lat <= 36)
ext = [lon[sx][0], lon[sx][-1], lat[sy][0], lat[sy][-1]]
fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.4), dpi=200,
                         gridspec_kw={"wspace": 0.06, "hspace": 0.18})
ls = LightSource(azdeg=315, altdeg=45)
ax = axes[0, 0]
zz = bathy[np.ix_(sy, sx)]
ax.imshow(ls.shade(zz, cmap=plt.cm.Greys, vert_exag=0.5, blend_mode="overlay"),
          extent=ext, origin="lower", aspect="auto")
ax.set_title("(a) 海底地形（阴影）", fontsize=8)
for ax, fld, title, sym in [(axes[0, 1], ssv, "(b) 1′ 重力异常 (mGal)", True),
                            (axes[1, 0], ckf, "(c) 15″ 重力异常 (mGal)", True),
                            (axes[1, 1], ckf-ssv, "(d) 15″−1′ (mGal)", False)]:
    z = fld[np.ix_(sy, sx)]
    v = np.nanpercentile(np.abs(z), 98)
    im = ax.imshow(z, cmap="RdBu_r", extent=ext, origin="lower", aspect="auto",
                   vmin=-v, vmax=v)
    ax.set_title(title, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.01).ax.tick_params(labelsize=5)
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
    ax.add_patch(plt.Rectangle((139.2, 34.6), 1.2, 0.9, fill=False,
                               ec="#C0392B", lw=1.2))
fig.savefig(OUT / "fig_new_paper_zoom4.png", bbox_inches="tight")
print("saved")
