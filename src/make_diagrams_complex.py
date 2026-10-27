"""复杂化第 1 页波长组成、第 6 页标准 KF 循环。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, RED, GOLD, LIGHT = "#0B2545", "#C0392B", "#C9A227", "#EEF3F9"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})


def concept():
    g = xr.open_dataset(ROOT / "data/ssv32/grav_33.1.nc")
    gv = g["z"].values[::20, ::20]
    glon = g["lon"].values[::20]; glat = g["lat"].values[::20]

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0), dpi=220,
                             gridspec_kw={"width_ratios": [1.1, 1.0], "wspace": 0.18})
    ax = axes[0]
    ax.set_facecolor("#E6E6E6")
    im = ax.imshow(gv, cmap="RdBu_r", origin="lower",
                   extent=[glon[0], glon[-1], glat[0], glat[-1]],
                   vmin=-100, vmax=100, aspect="auto")
    ax.set_title("(a) 全球海洋重力异常（SSV33.1，1′）\n本文使用的 1′ 产品",
                 fontsize=9.5, pad=4)
    ax.set_xlabel("经度 (°E)", fontsize=8); ax.set_ylabel("纬度 (°N)", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("mGal", fontsize=7)

    ax = axes[1]
    ax.set_xscale("log")
    ax.set_xlim(1, 1000); ax.set_ylim(40, -2)
    ax.set_xlabel("波长 (km)", fontsize=9); ax.set_ylabel("等效深度 (km)", fontsize=9)
    ax.set_title("(b) 波长—深度—构造", fontsize=11)
    ax.grid(alpha=0.2, which="both")
    ax.tick_params(labelsize=7)

    bands = [
        (100, 1000, 20, 40, "#D6E2F0", "地幔 / 地壳厚度"),
        (20, 100, 6, 20, "#F4D9B0", "海山 / 盆地"),
        (5, 20, 1.5, 6, "#F2B8B5", "断裂 / 海底地形"),
        (1, 5, 0.2, 1.5, "#C8E6C9", "浅部噪声"),
    ]
    for x0, x1, y0, y1, c, lab in bands:
        ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, fc=c, ec="white",
                                   alpha=0.9, lw=1))
        ax.text(np.sqrt(x0*x1), (y0+y1)/2, lab, ha="center", va="center",
                fontsize=8, color=NAVY)
    ax.add_patch(plt.Rectangle((5, 1.5), 15, 4.5, fill=False, ec=RED,
                               lw=1.6, ls="--"))
    ax.text(10, 38, "15″ 目标：5–20 km", ha="center", fontsize=8, color=RED)
    fig.savefig(OUT / "fig_new_concept_rich.png", bbox_inches="tight")
    plt.close(fig)


def kf_cycle():
    fig, ax = plt.subplots(figsize=(13.4, 5.2), dpi=220)
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    def node(x, y, w, h, title, lines, color=LIGHT):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.012,rounding_size=0.03",
                                    fc=color, ec=NAVY, lw=1.2))
        ax.text(x+w/2, y+h*0.76, title, ha="center", va="center", fontsize=12,
                color=NAVY, fontweight="bold")
        for i, t in enumerate(lines):
            ax.text(x+w/2, y+h*(0.48 - i*0.20), t, ha="center", va="center",
                    fontsize=8, color="#333333")

    node(0.32, 0.74, 0.36, 0.18, "① 预测", ["$x^-=Ax$", "$P^-=APA^{T}+Q$"])
    node(0.68, 0.40, 0.28, 0.18, "② 观测", ["$z=Hx+v$", "$R$：观测噪声"])
    node(0.32, 0.08, 0.36, 0.18, "④ 更新", ["$x=x^-+K(z-Hx^-)$", "$P=(I-KH)P^-$"])
    node(0.04, 0.40, 0.28, 0.18, "③ 增益", ["$K=P^-H^{T}$", "$(HP^-H^{T}+R)^{-1}$"])

    ax.add_patch(Circle((0.5, 0.50), 0.105, fc=NAVY, ec=GOLD, lw=2))
    ax.text(0.5, 0.53, "KF", ha="center", va="center", fontsize=18,
            color=GOLD, fontweight="bold")
    ax.text(0.5, 0.45, "递推", ha="center", va="center", fontsize=8, color="white")

    arrows = [
        ((0.68, 0.80), (0.68, 0.58), 0.25),
        ((0.68, 0.40), (0.68, 0.22), 0.25),
        ((0.32, 0.17), (0.18, 0.40), 0.25),
        ((0.18, 0.58), (0.32, 0.80), 0.25),
    ]
    for p0, p1, rad in arrows:
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=17,
                                     color=RED, lw=1.8,
                                     connectionstyle=f"arc3,rad={rad}"))
    ax.text(0.73, 0.63, "下一轮", fontsize=8, color=RED)

    ax.add_patch(FancyBboxPatch((0.04, 0.80), 0.20, 0.12,
                                boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc="#FFF8E7", ec=GOLD, lw=1))
    ax.text(0.14, 0.86, "输入\n卫星 / 船测", ha="center", va="center", fontsize=8)
    ax.add_patch(FancyBboxPatch((0.78, 0.08), 0.18, 0.12,
                                boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc="#FCEDEA", ec=RED, lw=1))
    ax.text(0.87, 0.14, "输出\n15″ 场 + P", ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(0.32, 0.82), xytext=(0.24, 0.86),
                arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=1.2))
    ax.annotate("", xy=(0.78, 0.14), xytext=(0.68, 0.16),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2))
    ax.text(0.5, 0.97, "标准卡尔曼滤波：预测—观测—增益—更新", ha="center",
            fontsize=13, color=NAVY, fontweight="bold")
    ax.text(0.5, 0.02, "权重由 R/Q 自动分配；P 给出逐像元方差",
            ha="center", fontsize=9, color=RED)
    fig.savefig(OUT / "fig_new_stdkf_cycle.png", bbox_inches="tight")
    plt.close(fig)
    print("saved concept/kf")


if __name__ == "__main__":
    concept()
    kf_cycle()
