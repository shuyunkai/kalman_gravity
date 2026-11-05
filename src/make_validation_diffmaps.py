"""空间精度差异图：CKF 减其他方法的绝对误差。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})

d = np.load(OUT / "validation_G_fields.npz")
lon, lat = d["lon"], d["lat"]
val = np.asarray(d["val"], float)
ix = np.clip(np.searchsorted(lon, val[:, 0]), 0, len(lon)-1)
iy = np.clip(np.searchsorted(lat, val[:, 1]), 0, len(lat)-1)
err = {k: np.abs(d[k][iy, ix] - val[:, 2]) for k in ("ssv", "freq", "lsc", "ckf")}
comps = [("双线性", "ssv"), ("频域融合", "freq"), ("LSC", "lsc")]

x0, x1, y0, y1 = 138, 147, 34, 42
m = (val[:, 0] >= x0) & (val[:, 0] <= x1) & (val[:, 1] >= y0) & (val[:, 1] <= y1)
sub = (slice(np.searchsorted(lat, y0), np.searchsorted(lat, y1)),
       slice(np.searchsorted(lon, x0), np.searchsorted(lon, x1)))
bg = d["ckf"][sub]
ext = [lon[sub[1]][0], lon[sub[1]][-1], lat[sub[0]][0], lat[sub[0]][-1]]
vmax = 2.0

fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.6), dpi=220)
for ax, (name, key) in zip(axes, comps):
    ax.imshow(bg, cmap="Greys", origin="lower", extent=ext, alpha=0.28,
              vmin=-100, vmax=100, aspect="auto")
    diff = (err["ckf"] - err[key])[m]
    sc = ax.scatter(val[m, 0], val[m, 1], c=diff, cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax, s=10, edgecolors="none")
    ax.set_title(f"相干保持 KF − {name}：均值 {diff.mean():+.2f} mGal",
                 fontsize=8.5)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xticks([]); ax.set_yticks([])
cb = fig.colorbar(sc, ax=axes, fraction=0.018, pad=0.012)
cb.set_label("误差差 (mGal)；蓝=相干保持 KF 更低", fontsize=7)
cb.ax.tick_params(labelsize=6)
fig.savefig(OUT / "fig_new_validation_diffmaps.png", bbox_inches="tight")
print("saved")
