"""第1页重排：a 全球场，b 波长叠加，右侧放傅里叶展开。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, RED, GOLD, BLUE, GRAY = "#0B2545", "#C0392B", "#C9A227", "#7FA8C9", "#90A4AE"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})

ds = xr.open_dataset(ROOT / "data/ssv32/grav_33.1.nc")
gv = ds["z"].values[::20, ::20]
glon = ds["lon"].values[::20]; glat = ds["lat"].values[::20]

fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.0), dpi=220,
                         gridspec_kw={"width_ratios": [1.35, 1.05, 0.95],
                                      "wspace": 0.12})
# (a) map
ax = axes[0]
im = ax.imshow(gv, cmap="RdBu_r", origin="lower",
               extent=[glon[0], glon[-1], glat[0], glat[-1]],
               vmin=-100, vmax=100, aspect="auto")
ax.set_title("(a) 全球海洋重力异常", fontsize=9, pad=3)
ax.set_xlabel("经度 (°E)", fontsize=7); ax.set_ylabel("纬度 (°N)", fontsize=7)
ax.tick_params(labelsize=6)
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("mGal", fontsize=6)

# (b) curves
ax = axes[1]
x = np.linspace(0, 1, 800)
long = 0.30*np.sin(2*np.pi*1.2*x)
mid = 0.16*np.sin(2*np.pi*7*x + 0.6)
short = 0.07*np.sin(2*np.pi*22*x + 1.1)
ax.plot(x, long, color=GRAY, lw=1.3, label="长波")
ax.plot(x, mid, color=BLUE, lw=1.2, label="中波")
ax.plot(x, short, color=RED, lw=1.1, label="短波")
ax.plot(x, long+mid+short, color=NAVY, lw=1.6, label="叠加")
ax.annotate("", xy=(0.307, 0.30), xytext=(0.165, 0.30),
            arrowprops=dict(arrowstyle="<->", color=RED, lw=1))
ax.text(0.236, 0.36, "λ = 相邻波峰距离", ha="center", fontsize=7,
        color=RED)
ax.set_xlim(0, 1); ax.set_ylim(-0.7, 0.7)
ax.set_title("(b) 不同波长成分叠加", fontsize=9)
ax.set_yticks([]); ax.tick_params(labelsize=6)
ax.legend(fontsize=6, frameon=False, ncol=2, loc="lower center")
ax.grid(alpha=0.15)

# (c) formulas
ax = axes[2]; ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.02, 0.94, "傅里叶展开", fontsize=10, color=GOLD, fontweight="bold")
ax.text(0.02, 0.84, r"$g=g_L+g_M+g_S$", fontsize=13, color=NAVY)
ax.text(0.02, 0.73, r"长波 $g_L=A_L\sin(2\pi x/\lambda_L+\phi_L)$",
        fontsize=7.5, color=GRAY)
ax.text(0.02, 0.62, r"中波 $g_M=A_M\sin(2\pi x/\lambda_M+\phi_M)$",
        fontsize=7.5, color=BLUE)
ax.text(0.02, 0.51, r"短波 $g_S=A_S\sin(2\pi x/\lambda_S+\phi_S)$",
        fontsize=7.5, color=RED)
ax.text(0.02, 0.38, "λ：相邻波峰距离", fontsize=8, color=RED)
ax.text(0.02, 0.28, "λ 越大，变化越慢；\nλ 越小，变化越快。", fontsize=8,
        color="#44546A")
ax.text(0.02, 0.14, "重力场 = 不同尺度成分的叠加", fontsize=7.5, color=NAVY)

fig.tight_layout()
fig.savefig(OUT / "fig_new_field_wave_detail.png", bbox_inches="tight")
print("saved")
