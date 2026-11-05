"""汇报式前两页：定义+公式+场；采样+分辨率+5-20km波段。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import xarray as xr

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, RED, GOLD, BLUE, GRAY = "#0B2545", "#C0392B", "#C9A227", "#7FA8C9", "#90A4AE"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})

ds = xr.open_dataset(ROOT / "data/ssv32/grav_33.1.nc")
gv = ds["z"].values[::20, ::20]
glon = ds["lon"].values[::20]; glat = ds["lat"].values[::20]


def field():
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.0), dpi=220,
                             gridspec_kw={"width_ratios": [0.9, 1.35, 1.05],
                                          "wspace": 0.12})
    ax = axes[0]; ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.02, 0.95, "重力场", fontsize=11, color=NAVY, fontweight="bold")
    ax.text(0.02, 0.86, r"$\Delta g=g_{obs}-\gamma$", fontsize=13, color=RED)
    ax.text(0.02, 0.79, r"$g_{obs}$：实测重力", fontsize=8, color="#44546A")
    ax.text(0.02, 0.73, r"$\gamma$：理论正常场", fontsize=8, color="#44546A")
    ax.text(0.02, 0.62, "按波长分解：", fontsize=9, color=GOLD, fontweight="bold")
    ax.text(0.02, 0.55, r"$g=g_L+g_M+g_S$", fontsize=11, color=NAVY)
    ax.text(0.02, 0.45, r"长波 $g_L=A_L\sin(2\pi x/\lambda_L+\phi_L)$",
            fontsize=7, color=GRAY)
    ax.text(0.02, 0.35, r"中波 $g_M=A_M\sin(2\pi x/\lambda_M+\phi_M)$",
            fontsize=7, color=BLUE)
    ax.text(0.02, 0.25, r"短波 $g_S=A_S\sin(2\pi x/\lambda_S+\phi_S)$",
            fontsize=7, color=RED)
    ax.text(0.02, 0.12, "λ：相邻波峰距离", fontsize=7.5, color=RED)

    ax = axes[1]
    im = ax.imshow(gv, cmap="RdBu_r", origin="lower",
                   extent=[glon[0], glon[-1], glat[0], glat[-1]],
                   vmin=-100, vmax=100, aspect="auto")
    ax.set_title("(a) 全球海洋重力异常", fontsize=9, pad=3)
    ax.set_xlabel("经度 (°E)", fontsize=7); ax.set_ylabel("纬度 (°N)", fontsize=7)
    ax.tick_params(labelsize=6)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("mGal", fontsize=6)

    ax = axes[2]
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

    fig.tight_layout()
    fig.savefig(OUT / "fig_new_field_wave_detail.png", bbox_inches="tight")


def resolution():
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.0), dpi=220,
                             gridspec_kw={"width_ratios": [0.8, 1.0, 1.2],
                                          "wspace": 0.12})
    ax = axes[0]; ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(Rectangle((0.10, 0.42), 0.52, 0.36, fill=False, ec=NAVY, lw=1.6))
    for i in range(1, 4):
        ax.plot([0.10+0.13*i, 0.10+0.13*i], [0.42, 0.78], color=NAVY, lw=0.7)
        ax.plot([0.10, 0.62], [0.42+0.09*i, 0.42+0.09*i], color=NAVY, lw=0.7)
    ax.text(0.36, 0.84, "1′ 与 15″ 格距", ha="center", fontsize=9,
            color=NAVY, fontweight="bold")
    ax.text(0.36, 0.35, "1′  Δx≈1.85 km", ha="center", fontsize=8, color=NAVY)
    ax.text(0.36, 0.28, "15″ Δx≈0.46 km", ha="center", fontsize=8, color=RED)
    ax.text(0.36, 0.17, "1′ 格内 4×4 个 15″ 格点", ha="center", fontsize=7,
            color="#44546A")

    ax = axes[1]
    ax.set_xlim(0, 1); ax.set_ylim(-1.15, 1.15)
    xx = np.linspace(0, 1, 600)
    ax.plot(xx, np.sin(2*np.pi*2*xx), color=NAVY, lw=1.3)
    xs = np.arange(0, 1.001, 0.25)
    ax.plot(xs, np.sin(2*np.pi*2*xs), "o", color=RED, ms=4)
    for xv in xs:
        ax.plot([xv, xv], [-1.1, 1.1], color=GRAY, lw=0.4, ls=":")
    ax.annotate("", xy=(0.5, 0.75), xytext=(0.0, 0.75),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1))
    ax.text(0.25, 0.9, "λ_min=2Δx", ha="center", fontsize=9, color=RED,
            fontweight="bold")
    ax.set_title("(b) 一个波长至少 2 个采样点", fontsize=9)
    ax.set_yticks([]); ax.tick_params(labelsize=7)

    ax = axes[2]
    ax.set_xscale("log"); ax.set_xlim(1, 100); ax.set_ylim(0, 1)
    ax.axvspan(5, 20, color=RED, alpha=0.12)
    ax.axvline(0.93, color=BLUE, lw=1.2, ls="--")
    ax.axvline(3.7, color=NAVY, lw=1.2, ls="--")
    ax.axvline(12, color=GRAY, lw=1.0, ls=":")
    ax.axvline(20, color=GRAY, lw=1.0, ls=":")
    ax.text(1.15, 0.62, "15″ λ_min≈0.93", fontsize=7, color=BLUE, rotation=90)
    ax.text(3.2, 0.62, "1′ λ_min≈3.7", fontsize=7, color=NAVY, rotation=90)
    ax.text(15, 0.12, "5–20 km 目标", ha="center", fontsize=8, color=RED)
    ax.text(12, 0.82, "1′ 产品有效 12–20 km", fontsize=7, color=GRAY)
    ax.annotate("5–12 km 被平滑", xy=(8, 0.35), xytext=(1.3, 0.35),
                fontsize=7.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
    ax.set_xlabel("波长 (km)", fontsize=8)
    ax.set_title("(c) 1′ 产品与 5–20 km 短波", fontsize=9)
    ax.set_yticks([]); ax.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(OUT / "fig_new_resolution_detail.png", bbox_inches="tight")


field()
resolution()
print("saved")
