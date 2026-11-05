"""第1页：重力场与波长；第2页仍用分辨率图。"""
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

fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0), dpi=220,
                         gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.10})
ax = axes[0]
im = ax.imshow(gv, cmap="RdBu_r", origin="lower",
               extent=[glon[0], glon[-1], glat[0], glat[-1]],
               vmin=-100, vmax=100, aspect="auto")
ax.set_title("(a) 全球海洋重力异常", fontsize=11, pad=4)
ax.set_xlabel("经度 (°E)", fontsize=8); ax.set_ylabel("纬度 (°N)", fontsize=8)
ax.tick_params(labelsize=7)
fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).set_label("mGal", fontsize=7)
ax.text(0.01, 0.02, "逐点 Δg = g − γ  →  Δg(x,y)", transform=ax.transAxes,
        fontsize=9, color=NAVY, fontweight="bold",
        bbox=dict(fc="white", ec="none", alpha=0.75, pad=2))

ax = axes[1]
x = np.linspace(0, 1, 800)
long = 0.30*np.sin(2*np.pi*1.2*x)
mid = 0.16*np.sin(2*np.pi*7*x + 0.6)
short = 0.07*np.sin(2*np.pi*22*x + 1.1)
ax.plot(x, long, color=GRAY, lw=1.4, label="长波")
ax.plot(x, mid, color=BLUE, lw=1.3, label="中波")
ax.plot(x, short, color=RED, lw=1.2, label="短波")
ax.plot(x, long+mid+short, color=NAVY, lw=1.8, label="叠加场")
ax.annotate("", xy=(0.307, 0.30), xytext=(0.165, 0.30),
            arrowprops=dict(arrowstyle="<->", color=RED, lw=1.0))
ax.text(0.236, 0.35, "波长 λ = 相邻波峰距离", ha="center", fontsize=9,
        color=RED, fontweight="bold")
ax.set_xlim(0, 1); ax.set_ylim(-0.7, 0.7)
ax.set_title("(b) 重力场 = 不同波长成分叠加", fontsize=10)
ax.set_xlabel(""); ax.set_yticks([])
ax.legend(fontsize=7, frameon=False, ncol=2, loc="lower center")
ax.grid(alpha=0.15)
ax.text(0.02, 0.93, r"$g(x)=\sum_k A_k\sin(2\pi x/\lambda_k+\phi_k)$",
        transform=ax.transAxes, fontsize=10, color=NAVY,
        bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
fig.tight_layout(rect=[0, 0.18, 1, 1])
fig.text(0.03, 0.025, r"长波 $g_L=A_L\sin(2\pi x/\lambda_L+\phi_L)$",
         fontsize=8, color=GRAY)
fig.text(0.36, 0.025, r"中波 $g_M=A_M\sin(2\pi x/\lambda_M+\phi_M)$",
         fontsize=8, color=BLUE)
fig.text(0.69, 0.025, r"短波 $g_S=A_S\sin(2\pi x/\lambda_S+\phi_S)$",
         fontsize=8, color=RED)
fig.savefig(OUT / "fig_new_field_wave.png", bbox_inches="tight")
print("saved")
