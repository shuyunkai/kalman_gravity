"""日本区：地形 + 双线性/频域/LSC/相干保持KF。"""
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
bathy = np.load(ROOT / "output_real/three_parts/ship_regions/G_Japan.npz")["bathy"]
sx = (lon >= 138) & (lon <= 147); sy = (lat >= 34) & (lat <= 42)
ext = [lon[sx][0], lon[sx][-1], lat[sy][0], lat[sy][-1]]
fields = [("双线性", d["ssv"]), ("频域融合", d["freq"]),
          ("LSC", d["lsc"]), ("相干保持 KF", d["ckf"])]
fig, axes = plt.subplots(1, 5, figsize=(13.4, 3.8), dpi=200,
                         gridspec_kw={"wspace": 0.04})
ls = LightSource(azdeg=315, altdeg=45)
z = bathy[np.ix_(sy, sx)]
axes[0].imshow(ls.shade(z, cmap=plt.cm.Greys, vert_exag=0.5, blend_mode="overlay"),
               origin="lower", extent=ext, aspect="auto")
axes[0].set_title("地形", fontsize=8)
for ax, (name, fld) in zip(axes[1:], fields):
    zz = fld[np.ix_(sy, sx)]
    v = np.nanpercentile(np.abs(zz), 98)
    ax.imshow(zz, cmap="RdBu_r", origin="lower", extent=ext,
              vmin=-v, vmax=v, aspect="auto")
    ax.set_title(name, fontsize=8)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.savefig(OUT / "fig_new_japan_methods.png", bbox_inches="tight")
print("saved")
