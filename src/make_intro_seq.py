"""前四页图：场、波长、分辨率、提高分辨率的本质。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import xarray as xr

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, RED, GOLD, BLUE, GRAY = "#0B2545", "#C0392B", "#C9A227", "#7FA8C9", "#90A4AE"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})


def field():
    ds = xr.open_dataset(ROOT / "data/ssv32/grav_33.1.nc")
    gv = ds["z"].values[::20, ::20]
    glon = ds["lon"].values[::20]; glat = ds["lat"].values[::20]
    fig, ax = plt.subplots(figsize=(13.4, 5.0), dpi=220)
    im = ax.imshow(gv, cmap="RdBu_r", origin="lower",
                   extent=[glon[0], glon[-1], glat[0], glat[-1]],
                   vmin=-100, vmax=100, aspect="auto")
    ax.set_title("全球海洋重力异常（SSV33.1，1′）", fontsize=12, pad=6)
    ax.set_xlabel("经度 (°E)", fontsize=9); ax.set_ylabel("纬度 (°N)", fontsize=9)
    ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).set_label("mGal", fontsize=8)
    ax.text(0.01, 0.02, "逐点 Δg = g − γ  →  Δg(x,y)", transform=ax.transAxes,
            fontsize=10, color=NAVY, fontweight="bold",
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=2))
    fig.tight_layout()
    fig.savefig(OUT / "fig_new_field_only.png", bbox_inches="tight")


def wavelength():
    fig, ax = plt.subplots(figsize=(13.4, 5.0), dpi=220)
    x = np.linspace(0, 1, 800)
    long = 0.30*np.sin(2*np.pi*1.2*x)
    mid = 0.16*np.sin(2*np.pi*7*x + 0.6)
    short = 0.07*np.sin(2*np.pi*22*x + 1.1)
    ax.plot(x, long, color=GRAY, lw=1.6, label="长波")
    ax.plot(x, mid, color=BLUE, lw=1.5, label="中波")
    ax.plot(x, short, color=RED, lw=1.4, label="短波")
    ax.plot(x, long+mid+short, color=NAVY, lw=2.0, label="叠加场")
    ax.annotate("", xy=(0.307, 0.55), xytext=(0.165, 0.55),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.2))
    ax.text(0.236, 0.59, "波长 λ = 相邻波峰距离", ha="center", fontsize=11,
            color=RED, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(-0.7, 0.7)
    ax.set_title("重力场 = 不同波长成分的叠加", fontsize=12)
    ax.set_xlabel("位置", fontsize=9); ax.set_yticks([])
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="lower center")
    ax.grid(alpha=0.15)
    ax.text(0.01, 0.02, "长波：变化慢", transform=ax.transAxes,
            fontsize=8.5, color="#44546A")
    ax.text(0.99, 0.02, "短波：变化快", transform=ax.transAxes,
            fontsize=8.5, color="#44546A", ha="right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_new_wavelength_only.png", bbox_inches="tight")


def resolution():
    fig, ax = plt.subplots(figsize=(13.4, 5.0), dpi=220)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(Rectangle((0.08, 0.38), 0.34, 0.40, fill=False, ec=NAVY, lw=2))
    for i in range(1, 4):
        ax.plot([0.08+0.085*i, 0.08+0.085*i], [0.38, 0.78], color=NAVY, lw=0.8)
        ax.plot([0.08, 0.42], [0.38+0.10*i, 0.38+0.10*i], color=NAVY, lw=0.8)
    ax.text(0.25, 0.83, "1′ 格网 = 4×4 个 15″ 格网", ha="center", fontsize=11,
            color=NAVY, fontweight="bold")
    ax.text(0.05, 0.30, "1′  Δx ≈ 1.85 km", fontsize=9, color=NAVY)
    ax.text(0.05, 0.23, "15″  Δx ≈ 0.46 km", fontsize=9, color=NAVY)
    ax.text(0.55, 0.70, "网格间距 Δx", fontsize=11, color=GOLD, fontweight="bold")
    ax.text(0.55, 0.60, "决定能采到的最短波长", fontsize=9, color="#44546A")
    ax.text(0.55, 0.48, "λ_min = 2Δx", fontsize=16, color=RED, fontweight="bold")
    ax.text(0.55, 0.36, "1′：λ_min ≈ 3.7 km", fontsize=9, color="#44546A")
    ax.text(0.55, 0.28, "15″：λ_min ≈ 0.93 km", fontsize=9, color="#44546A")
    ax.text(0.55, 0.16, "小于 λ_min 的波长，网格采不到", fontsize=9,
            color=RED)
    ax.text(0.55, 0.08, "这就是分辨率", fontsize=9, color=RED, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_new_resolution_only.png", bbox_inches="tight")


def essence():
    fig, ax = plt.subplots(figsize=(13.4, 5.0), dpi=220)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # one 1' block, 4x4 15" unknowns
    ax.add_patch(Rectangle((0.06, 0.30), 0.34, 0.40, fill=False, ec=NAVY, lw=2))
    for i in range(1, 4):
        ax.plot([0.06+0.085*i, 0.06+0.085*i], [0.30, 0.70], color=NAVY, lw=0.7)
        ax.plot([0.06, 0.40], [0.30+0.10*i, 0.30+0.10*i], color=NAVY, lw=0.7)
    for i in range(4):
        for j in range(4):
            ax.text(0.1025+0.085*i, 0.35+0.10*j, "?", ha="center", va="center",
                    fontsize=8, color=RED)
    ax.text(0.23, 0.75, "1′ 块平均 = 1 个观测", ha="center", fontsize=10,
            color=NAVY, fontweight="bold")
    ax.text(0.23, 0.20, "块内 16 个 15″ 未知", ha="center", fontsize=9,
            color=RED)
    ax.text(0.50, 0.55, r"$\bar g_{1'}=\frac{1}{16}\sum g_{15''}$",
            fontsize=15, color=NAVY)
    ax.text(0.50, 0.42, "1 个方程，16 个未知", fontsize=10, color=RED,
            fontweight="bold")
    # constraints boxes
    for y, txt in [(0.72, "船测短波"), (0.58, "地形高频"), (0.44, "先验/平滑")]:
        ax.add_patch(Rectangle((0.66, y), 0.28, 0.10, fc="#FFF8E7", ec=GOLD))
        ax.text(0.80, y+0.05, txt, ha="center", va="center", fontsize=9, color=NAVY)
        ax.annotate("", xy=(0.50, y+0.05), xytext=(0.66, y+0.05),
                    arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.1))
    ax.text(0.80, 0.86, "外部约束", ha="center", fontsize=10, color=GOLD,
            fontweight="bold")
    ax.text(0.50, 0.10, "提高分辨率 = 用观测约束反演块内变化，不是简单插值",
            ha="center", fontsize=10, color=RED, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_new_highres_essence.png", bbox_inches="tight")


for f in (field, wavelength, resolution, essence):
    f()
print("saved")
