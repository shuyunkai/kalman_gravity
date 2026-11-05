"""空间精度页：四种方法 + 留出船测点误差着色。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, RED, GRAY, BLUE, GOLD = "#0B2545", "#C0392B", "#90A4AE", "#7FA8C9", "#C9A227"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})

d = np.load(OUT / "validation_G_fields.npz")
lon, lat = d["lon"], d["lat"]
val = np.asarray(d["val"], float)
ix = np.clip(np.searchsorted(lon, val[:, 0]), 0, len(lon)-1)
iy = np.clip(np.searchsorted(lat, val[:, 1]), 0, len(lat)-1)
methods = [("双线性", "ssv"), ("频域融合", "freq"), ("LSC", "lsc"),
           ("相干保持 KF", "ckf")]
errs = {name: np.abs(d[key][iy, ix] - val[:, 2]) for name, key in methods}
vmax = np.nanpercentile(np.concatenate([e for e in errs.values()]), 95)

x0, x1 = 138, 147
y0, y1 = 34, 42
sub = (slice(np.searchsorted(lat, y0), np.searchsorted(lat, y1)),
       slice(np.searchsorted(lon, x0), np.searchsorted(lon, x1)))
bg = d["ckf"][sub]
ext = [lon[sub[1]][0], lon[sub[1]][-1], lat[sub[0]][0], lat[sub[0]][-1]]
m = (val[:, 0] >= x0) & (val[:, 0] <= x1) & (val[:, 1] >= y0) & (val[:, 1] <= y1)

fig, axes = plt.subplots(2, 2, figsize=(13.4, 5.3), dpi=220)
for ax, (name, key) in zip(axes.ravel(), methods):
    ax.imshow(bg, cmap="RdBu_r", origin="lower", extent=ext, alpha=0.45,
              vmin=-100, vmax=100, aspect="auto")
    sc = ax.scatter(val[m, 0], val[m, 1], c=errs[name][m], cmap="RdYlGn_r",
                    vmin=0, vmax=vmax, s=9, edgecolors="none")
    rmse = float(np.sqrt(np.mean(errs[name][m]**2)))
    ax.set_title(f"{name}：RMSE={rmse:.2f} mGal", fontsize=9)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.tick_params(labelsize=6)
    ax.set_xticks([]); ax.set_yticks([])

cb = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.015)
cb.set_label("|模型 − 船测| (mGal)", fontsize=7)
cb.ax.tick_params(labelsize=6)
fig.savefig(OUT / "fig_new_validation_errormaps.png", bbox_inches="tight")
print("saved")
