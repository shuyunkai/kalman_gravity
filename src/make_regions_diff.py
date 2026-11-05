"""不同区域：1′、15″、差值 3×3 地图。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 8})
regions = [("D_Nankai", "南海海槽"), ("F_PhilSea", "菲律宾海"), ("G_Japan", "日本海沟")]
fig, axes = plt.subplots(3, 3, figsize=(13.4, 5.8), dpi=200,
                         gridspec_kw={"wspace": 0.04, "hspace": 0.18})
for i, (key, cn) in enumerate(regions):
    d = np.load(ROOT / f"output_real/three_parts/ship_regions/{key}.npz")
    lon, lat, ssv, kf = d["lon"], d["lat"], d["ssv"], d["kf"]
    step = max(1, min(ssv.shape)//400)
    ssv = ssv[::step, ::step]; kf = kf[::step, ::step]
    lon = lon[::step]; lat = lat[::step]
    ext = [lon[0], lon[-1], lat[0], lat[-1]]
    v = np.nanpercentile(np.abs(kf), 98)
    for j, (z, title) in enumerate([(ssv, "1′"), (kf, "15″"), (kf-ssv, "15″−1′")]):
        ax = axes[i, j]
        if j < 2:
            im = ax.imshow(z, cmap="RdBu_r", origin="lower", extent=ext,
                           vmin=-v, vmax=v, aspect="auto")
        else:
            vd = np.nanpercentile(np.abs(z), 98)
            im = ax.imshow(z, cmap="RdBu_r", origin="lower", extent=ext,
                           vmin=-vd, vmax=vd, aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_title(title, fontsize=9)
        if j == 0:
            ax.set_ylabel(cn, fontsize=8)
fig.savefig(OUT / "fig_new_regions_diff.png", bbox_inches="tight")
print("saved")
