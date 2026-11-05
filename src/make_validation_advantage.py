"""空间精度优势页：CKF 误差图 + 相对其他方法的 RMSE 降低百分比。"""
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
keys = {"双线性": "ssv", "频域融合": "freq", "LSC": "lsc", "相干保持 KF": "ckf"}
err = {k: np.abs(d[v][iy, ix] - val[:, 2]) for k, v in keys.items()}
rmse = {k: float(np.sqrt(np.mean(e**2))) for k, e in err.items()}

x0, x1, y0, y1 = 138, 147, 34, 42
m = (val[:, 0] >= x0) & (val[:, 0] <= x1) & (val[:, 1] >= y0) & (val[:, 1] <= y1)
sub = (slice(np.searchsorted(lat, y0), np.searchsorted(lat, y1)),
       slice(np.searchsorted(lon, x0), np.searchsorted(lon, x1)))
bg = d["ckf"][sub]
ext = [lon[sub[1]][0], lon[sub[1]][-1], lat[sub[0]][0], lat[sub[0]][-1]]

fig = plt.figure(figsize=(13.4, 5.0), dpi=220)
ax = fig.add_axes([0.03, 0.08, 0.58, 0.82])
ax.imshow(bg, cmap="Greys", origin="lower", extent=ext, alpha=0.3,
          vmin=-100, vmax=100, aspect="auto")
sc = ax.scatter(val[m, 0], val[m, 1], c=err["相干保持 KF"][m],
                cmap="RdYlGn_r", vmin=0, vmax=8, s=12, edgecolors="none")
ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"相干保持 KF：RMSE={rmse['相干保持 KF']:.2f} mGal", fontsize=9)
cb = fig.colorbar(sc, ax=ax, fraction=0.033, pad=0.015)
cb.set_label("|模型−船测| (mGal)", fontsize=7); cb.ax.tick_params(labelsize=6)

ax = fig.add_axes([0.68, 0.18, 0.29, 0.66])
names = ["双线性", "频域融合", "LSC"]
improve = [(1 - rmse["相干保持 KF"]/rmse[n])*100 for n in names]
ax.barh(names, improve, color=["#90A4AE", "#1F77B4", "#2CA02C"])
for i, v in enumerate(improve):
    ax.text(v+0.3, i, f"{v:.1f}%", va="center", fontsize=8, color="#1A1A1A")
ax.set_xlim(0, max(improve)*1.35)
ax.set_xlabel("RMSE 降低 (%)", fontsize=8)
ax.set_title("相对其他方法：均更低", fontsize=9)
ax.grid(axis="x", alpha=0.2); ax.tick_params(labelsize=7)

fig.savefig(OUT / "fig_new_validation_advantage.png", bbox_inches="tight")
print("saved")
