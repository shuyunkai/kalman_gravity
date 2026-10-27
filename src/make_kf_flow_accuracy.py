"""KF 完整流程图 + 多区域多方法精度对比图。"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, RED, GOLD, LIGHT = "#0B2545", "#C0392B", "#C9A227", "#EEF3F9"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})


def box(ax, x, y, w, h, title, sub="", fc="white", tc=NAVY, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=NAVY, lw=1.1))
    ax.text(x+w/2, y+h*0.64, title, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold")
    if sub:
        ax.text(x+w/2, y+h*0.30, sub, ha="center", va="center", fontsize=fs-1.5,
                color="#5B6573")


def arr(ax, p0, p1, rad=0.0, color="#5B6573"):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3,
                                connectionstyle=f"arc3,rad={rad}"))


def flow():
    fig, ax = plt.subplots(figsize=(13.4, 5.2), dpi=220)
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax.text(0.13, 0.95, "输入与预处理", ha="center", fontsize=12, color=GOLD,
            fontweight="bold")
    ax.text(0.48, 0.95, "迭代 KF（6 轮）", ha="center", fontsize=12, color=GOLD,
            fontweight="bold")
    ax.text(0.84, 0.95, "输出与验证", ha="center", fontsize=12, color=GOLD,
            fontweight="bold")

    box(ax, 0.02, 0.72, 0.20, 0.14, "卫星 1′", "SSV33.1 / DTU21")
    box(ax, 0.02, 0.53, 0.20, 0.14, "船测重力", "JODC / 航次")
    box(ax, 0.02, 0.34, 0.20, 0.14, "地形 GEBCO", "15″ 水深")
    box(ax, 0.26, 0.52, 0.18, 0.18, "块平均 + 先验", "1′ 块观测\nSSV + 地形高频", fc="#FFF8E7")
    for y in (0.79, 0.60, 0.41):
        arr(ax, (0.22, y), (0.26, 0.61))

    box(ax, 0.48, 0.72, 0.16, 0.13, "预测", "$x^-$、$P^-$", fc=LIGHT)
    box(ax, 0.68, 0.72, 0.16, 0.13, "卫星更新", "块观测 H\n标量增益", fc=LIGHT)
    box(ax, 0.68, 0.50, 0.16, 0.13, "船测更新", "点更新 + 邻域", fc=LIGHT)
    box(ax, 0.48, 0.50, 0.16, 0.13, "6 轮迭代", "修正量收敛", fc=LIGHT)
    arr(ax, (0.64, 0.785), (0.68, 0.785))
    arr(ax, (0.76, 0.72), (0.76, 0.63))
    arr(ax, (0.68, 0.565), (0.64, 0.565))
    arr(ax, (0.56, 0.63), (0.56, 0.72))
    ax.text(0.56, 0.675, "下一轮", ha="center", fontsize=7, color=RED)
    ax.text(0.76, 0.675, "R/Q", ha="center", fontsize=7, color=RED)

    box(ax, 0.48, 0.24, 0.36, 0.13, "RTS 反向平滑", "输出 $x_{smooth}$、$P_{smooth}$",
        fc="#FFF8E7")
    arr(ax, (0.56, 0.50), (0.56, 0.37))
    box(ax, 0.48, 0.06, 0.17, 0.12, "15″ 重力场", "1.85 km")
    box(ax, 0.67, 0.06, 0.17, 0.12, "逐像元方差", "$\\sqrt{P}$")
    arr(ax, (0.52, 0.24), (0.52, 0.18))
    arr(ax, (0.66, 0.24), (0.72, 0.18))

    box(ax, 0.86, 0.52, 0.12, 0.14, "留出验证", "整条航次", fc="#FCEDEA")
    box(ax, 0.86, 0.30, 0.12, 0.14, "精度指标", "RMSE / CDF", fc="#FCEDEA")
    arr(ax, (0.84, 0.30), (0.86, 0.59))
    arr(ax, (0.86, 0.52), (0.86, 0.44))
    ax.text(0.5, 0.005, "先验：$x_0$ = SSV 块平均 + α·地形布格高频；$P_0$、$Q$、$R$ 固定；"
            "相干保持预测：$x^-=(1-wC^2)G_\\sigma x+wC^2x$",
            ha="center", fontsize=8, color=RED)
    fig.savefig(OUT / "fig_new_kf_flow_full.png", bbox_inches="tight")
    plt.close(fig)


def accuracy():
    res = json.loads((OUT / "regions_coherence.json").read_text(encoding="utf-8"))
    keys = ["D_Nankai", "H_Kuroshio", "G_Japan"]
    names = ["南海-琉球", "九州东侧", "日本"]
    bil = [res[k]["bilinear"] for k in keys]
    freq = [res[k]["freq"] for k in keys]
    lsc = [res[k]["lsc"] for k in keys]
    ckf = [res[k]["ckf"] for k in keys]
    best = [min(a, b, c) for a, b, c in zip(bil, freq, lsc)]
    imp = [(1 - c / b) * 100 for b, c in zip(best, ckf)]

    fig = plt.figure(figsize=(13.4, 5.6), dpi=220)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.30, wspace=0.22)

    ax = fig.add_subplot(gs[0, :]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("(a) 三区 × 四种 15″ 场 RMSE（mGal）", fontsize=11, pad=2)
    cols = ["区域", "双线性", "频域融合", "LSC", "相干保持 KF", "改善（相对最优）"]
    widths = [0.14, 0.17, 0.17, 0.15, 0.18, 0.19]
    y = 1.0; h = 0.22
    x = 0
    for c, w in zip(cols, widths):
        ax.add_patch(Rectangle((x, y-h), w, h, fc=NAVY, ec="white", lw=1))
        ax.text(x+w/2, y-h/2, c, ha="center", va="center", color="white",
                fontsize=8.3, fontweight="bold")
        x += w
    for i, (n, b1, f1, l1, c1, im) in enumerate(zip(names, bil, freq, lsc, ckf, imp)):
        yy = y - h*(i+2)
        vals = [n, f"{b1:.2f}", f"{f1:.2f}", f"{l1:.2f}", f"{c1:.2f}", f"{im:.1f}%"]
        x = 0
        for val, w in zip(vals, widths):
            ax.add_patch(Rectangle((x, yy), w, h, fc=LIGHT, ec="white", lw=1))
            ax.text(x+w/2, yy+h/2, val, ha="center", va="center",
                    fontsize=8.6, color="#1A1A1A")
            x += w
    ax.text(0.0, 0.02, "三种常用场均以 SSV33.1 的 1′ 数据为输入；改善相对三者中最低 RMSE",
            fontsize=7.8, color=RED)

    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(3); w = 0.19
    ax.bar(x-1.5*w, bil, w, color="#D6E2F0", label="双线性")
    ax.bar(x-0.5*w, freq, w, color="#7FA8C9", label="频域融合")
    ax.bar(x+0.5*w, lsc, w, color=GOLD, label="LSC")
    ax.bar(x+1.5*w, ckf, w, color=NAVY, label="相干保持 KF")
    for i in range(3):
        ax.text(x[i]+1.5*w, ckf[i]+0.08, f"{ckf[i]:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("RMSE (mGal)", fontsize=9)
    ax.set_title("(b) 分组对比", fontsize=11)
    ax.set_xlim(-0.65, 2.65)
    ax.legend(fontsize=6.5, frameon=False, ncol=2); ax.grid(axis="y", alpha=0.2)
    ax.tick_params(labelsize=7.5)

    ax = fig.add_subplot(gs[1, 1])
    d = np.load(OUT / "validation_G_fields.npz")
    lon, lat = d["lon"], d["lat"]
    val = np.asarray(d["val"], float)
    ix = np.clip(np.searchsorted(lon, val[:, 0]), 0, len(lon)-1)
    iy = np.clip(np.searchsorted(lat, val[:, 1]), 0, len(lat)-1)
    for name, fld, col in (("双线性", d["ssv"], "#D6E2F0"),
                           ("相干保持 KF", d["ckf"], NAVY)):
        e = np.abs(fld[iy, ix] - val[:, 2])
        ax.plot(np.sort(e), np.linspace(0, 1, len(e)), color=col, lw=1.6, label=name)
    ax.set_xlim(0, 8); ax.set_xlabel("|误差| (mGal)", fontsize=9)
    ax.set_ylabel("累积比例", fontsize=9)
    ax.set_title("(c) 误差 CDF（双线性 vs CKF）", fontsize=10)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
    ax.tick_params(labelsize=8)
    fig.savefig(OUT / "fig_new_accuracy_table.png", bbox_inches="tight")
    print("saved flow/accuracy")


if __name__ == "__main__":
    flow()
    accuracy()
