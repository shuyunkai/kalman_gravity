"""第 1 页：重力场空间分布 + 异常正负与密度。"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, RED, GOLD, BLUE = "#0B2545", "#C0392B", "#C9A227", "#7FA8C9"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})

g = np.loadtxt(ROOT / "output_real/three_parts/newdeck/_none.txt") if False else None
import xarray as xr
ds = xr.open_dataset(ROOT / "data/ssv32/grav_33.1.nc")
gv = ds["z"].values[::20, ::20]
glon = ds["lon"].values[::20]; glat = ds["lat"].values[::20]

fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0), dpi=220,
                         gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.18})
ax = axes[0]
ax.set_facecolor("#E6E6E6")
im = ax.imshow(gv, cmap="RdBu_r", origin="lower",
               extent=[glon[0], glon[-1], glat[0], glat[-1]],
               vmin=-100, vmax=100, aspect="auto")
ax.set_title("(a) 全球海洋重力异常（SSV33.1，1′）", fontsize=11, pad=4)
ax.set_xlabel("经度 (°E)", fontsize=8); ax.set_ylabel("纬度 (°N)", fontsize=8)
ax.tick_params(labelsize=7)
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("mGal", fontsize=7)
ax.annotate("正异常", xy=(150, 20), xytext=(110, 45), color=RED, fontsize=8,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1))
ax.annotate("负异常", xy=(-160, 10), xytext=(-120, -35), color="#2874A6", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#2874A6", lw=1))

ax = axes[1]
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("(b) 异常正负反映地下密度差异", fontsize=10)
# water / crust
ax.add_patch(Rectangle((0.04, 0.55), 0.92, 0.08, fc="#DCEEF8", ec="none"))
ax.add_patch(Rectangle((0.04, 0.15), 0.92, 0.40, fc="#F1E2C8", ec="none"))
ax.text(0.05, 0.58, "海水", fontsize=7, color=BLUE)
ax.text(0.05, 0.17, "地壳", fontsize=7, color="#7A5C3A")
# density bodies
ax.add_patch(Rectangle((0.12, 0.25), 0.18, 0.20, fc="#E6B0AA", ec=RED, lw=1.2))
ax.text(0.21, 0.35, "高密度", ha="center", va="center", fontsize=8, color=RED)
ax.add_patch(Rectangle((0.62, 0.25), 0.18, 0.20, fc="#AED6F1", ec="#2874A6", lw=1.2))
ax.text(0.71, 0.35, "低密度", ha="center", va="center", fontsize=8, color="#2874A6")
# anomaly curves
x = np.linspace(0.04, 0.96, 400)
y = 0.82 + 0.08*np.exp(-((x-0.21)/0.075)**2) - 0.08*np.exp(-((x-0.71)/0.085)**2)
ax.plot(x, y, color=NAVY, lw=1.5)
ax.axhline(0.82, color="gray", lw=0.7, ls="--")
ax.annotate("Δg > 0", xy=(0.21, 0.90), xytext=(0.10, 0.94), color=RED,
            fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1))
ax.annotate("Δg < 0", xy=(0.71, 0.74), xytext=(0.78, 0.94), color="#2874A6",
            fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#2874A6", lw=1))
ax.text(0.5, 0.06, "同一位置：密度偏高→正异常；密度偏低→负异常", ha="center",
        fontsize=7.5, color="#44546A")

fig.tight_layout()
fig.savefig(OUT / "fig_new_concept_rich.png", bbox_inches="tight")
print("saved")
