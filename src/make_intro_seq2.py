"""前三页图：数值场、傅里叶分解、采样与分辨率。"""
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


def field():
    ds = xr.open_dataset(ROOT / "data/ssv32/grav_33.1.nc")
    gv = ds["z"].values[::20, ::20]
    glon = ds["lon"].values[::20]; glat = ds["lat"].values[::20]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0), dpi=220,
                             gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.10})
    ax = axes[0]
    im = ax.imshow(gv, cmap="RdBu_r", origin="lower",
                   extent=[glon[0], glon[-1], glat[0], glat[-1]],
                   vmin=-100, vmax=100, aspect="auto")
    ax.set_title("(a) 全球海洋重力异常（SSV33.1，1′）", fontsize=11, pad=4)
    ax.set_xlabel("经度 (°E)", fontsize=8); ax.set_ylabel("纬度 (°N)", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).set_label("mGal", fontsize=7)
    ax.text(0.01, 0.02, "逐点 Δg = g − γ  →  Δg(x,y)", transform=ax.transAxes,
            fontsize=9, color=NAVY, fontweight="bold",
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=2))

    ax = axes[1]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("(b) 连续场的离散采样", fontsize=10)
    vals = np.array([[-42, -15, 18, 35], [-30, -6, 22, 48],
                     [-12, 8, 31, 55], [5, 20, 40, 62]])
    for i in range(4):
        for j in range(4):
            ax.add_patch(Rectangle((0.08+j*0.18, 0.42+i*0.13), 0.16, 0.11,
                                   fc=plt.cm.RdBu_r((vals[i, j]+80)/160),
                                   ec="white", lw=0.8))
            ax.text(0.16+j*0.18, 0.475+i*0.13, f"{vals[i,j]:d}",
                    ha="center", va="center", fontsize=7, color="black")
    ax.text(0.45, 0.36, "每个格点 = 一个 Δg 采样值", ha="center", fontsize=8.5,
            color=NAVY)
    ax.text(0.45, 0.27, "格点越密 → 越接近连续场", ha="center", fontsize=9,
            color=RED, fontweight="bold")
    ax.text(0.45, 0.18, "不是把点拼起来，而是场的离散表示", ha="center", fontsize=8,
            color="#44546A")
    fig.tight_layout()
    fig.savefig(OUT / "fig_new_field_only.png", bbox_inches="tight")


def wavelength():
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0), dpi=220,
                             gridspec_kw={"width_ratios": [1.5, 0.9], "wspace": 0.10})
    ax = axes[0]
    x = np.linspace(0, 1, 800)
    long = 0.30*np.sin(2*np.pi*1.2*x)
    mid = 0.16*np.sin(2*np.pi*7*x + 0.6)
    short = 0.07*np.sin(2*np.pi*22*x + 1.1)
    ax.plot(x, long, color=GRAY, lw=1.5, label="长波")
    ax.plot(x, mid, color=BLUE, lw=1.4, label="中波")
    ax.plot(x, short, color=RED, lw=1.3, label="短波")
    ax.plot(x, long+mid+short, color=NAVY, lw=1.9, label="叠加场")
    ax.annotate("", xy=(0.307, 0.55), xytext=(0.165, 0.55),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.1))
    ax.text(0.236, 0.59, "波长 λ = 相邻波峰距离", ha="center", fontsize=9,
            color=RED, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(-0.7, 0.7)
    ax.set_title("(a) 重力场 = 不同波长成分的叠加", fontsize=10)
    ax.set_xlabel("位置", fontsize=8); ax.set_yticks([])
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="lower center")
    ax.grid(alpha=0.15)

    ax = axes[1]; ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("(b) 分解原理", fontsize=10)
    ax.text(0.05, 0.80, "任意空间曲线", fontsize=9, color=NAVY)
    ax.text(0.05, 0.72, "可写成不同波长正弦之和：", fontsize=9, color=NAVY)
    ax.text(0.05, 0.52, r"$g(x)=\sum_k A_k\sin(2\pi x/\lambda_k+\phi_k)$",
            fontsize=12, color=NAVY)
    ax.text(0.05, 0.34, "长波：λ 大，变化慢", fontsize=8.5, color=GRAY)
    ax.text(0.05, 0.26, "短波：λ 小，变化快", fontsize=8.5, color=RED)
    ax.text(0.05, 0.12, "分解后，每个波长成分对应一个空间尺度", fontsize=8,
            color="#44546A")
    fig.tight_layout()
    fig.savefig(OUT / "fig_new_wavelength_only.png", bbox_inches="tight")


def resolution():
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0), dpi=220,
                             gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.10})
    ax = axes[0]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(Rectangle((0.10, 0.42), 0.36, 0.38, fill=False, ec=NAVY, lw=1.8))
    for i in range(1, 4):
        ax.plot([0.10+0.09*i, 0.10+0.09*i], [0.42, 0.80], color=NAVY, lw=0.7)
        ax.plot([0.10, 0.46], [0.42+0.095*i, 0.42+0.095*i], color=NAVY, lw=0.7)
    ax.text(0.28, 0.85, "1′ 格网 = 4×4 个 15″ 格网", ha="center", fontsize=9,
            color=NAVY, fontweight="bold")
    ax.text(0.28, 0.35, "1′ Δx ≈ 1.85 km", ha="center", fontsize=8, color=NAVY)
    ax.text(0.28, 0.28, "15″ Δx ≈ 0.46 km", ha="center", fontsize=8, color=NAVY)

    ax = axes[1]
    ax.set_xlim(0, 1); ax.set_ylim(-1.1, 1.1)
    x = np.linspace(0, 1, 600)
    ax.plot(x, np.sin(2*np.pi*2*x), color=NAVY, lw=1.4)
    xs = np.arange(0, 1.001, 0.25)
    ax.plot(xs, np.sin(2*np.pi*2*xs), "o", color=RED, ms=4)
    for xv in xs:
        ax.plot([xv, xv], [-1.05, 1.05], color=GRAY, lw=0.4, ls=":")
    ax.set_title("(b) 采样：一个波长至少 2 个点", fontsize=10)
    ax.set_xlabel("位置", fontsize=8); ax.set_yticks([])
    ax.annotate("", xy=(0.5, 0.75), xytext=(0.0, 0.75),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.0))
    ax.text(0.25, 0.88, "λ_min = 2Δx", ha="center", fontsize=10, color=RED,
            fontweight="bold")
    ax.text(0.5, -0.95, "采样间隔 Δx 越小，能采到的最短波长越短", ha="center",
            fontsize=8, color="#44546A")
    fig.tight_layout()
    fig.savefig(OUT / "fig_new_resolution_only.png", bbox_inches="tight")


for f in (field, wavelength, resolution):
    f()
print("saved")
