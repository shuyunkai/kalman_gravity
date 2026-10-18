"""验证部分新增两图：验证设计图 / 改善幅度图。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "savefig.facecolor": "white",
})
NAVY, MID, LIGHT, GREY, RED = "#16365C", "#4A7AA7", "#D8E0EC", "#5B6573", "#C0392B"
INK, PALE = "#1C2430", "#EEF3FA"
NODE = {"blue": ("#DCE6F5", "#9DB6DC", "#16365C"),
        "green": ("#DCEBDC", "#9CC79B", INK),
        "purple": ("#E4DEF0", "#B7A9D6", INK),
        "yellow": ("#FBF1CE", "#E0C97A", INK),
        "pink": ("#F8DEE2", "#E3A7B0", INK),
        "neutral": ("#EEF1F5", "#CAD2DE", INK)}
OUT = Path(r"D:\gravity\ppt_jq\figs")


def _canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, w * 1.03); ax.set_ylim(0, h * 1.03); ax.axis("off")
    return fig, ax


# ------------------------------------------------ 验证设计
def fig_validation_design():
    fig, ax = _canvas(11.4, 3.9)
    # 左：研究区图
    ax2 = fig.add_axes([0.03, 0.10, 0.42, 0.78])
    z = np.load(OUT / "_grav_cache.npy")
    ax2.imshow(z, extent=[124, 153, 12, 43], cmap="Blues_r", origin="lower",
               interpolation="nearest", aspect="auto")
    regions = [("G 综合验证区", 132, 26, 20, 16, "108 103 点"),
               ("D 南海-琉球", 124, 24, 8, 7, "12 310 点"),
               ("F 菲律宾海中部", 139, 13, 13, 13, "6 459 点")]
    for name, x0, y0, w, h, cnt in regions:
        ax2.add_patch(mp.Rectangle((x0, y0), w, h, fill=False, ec=RED, lw=1.8, zorder=5))
        ax2.text(x0 + 0.5, y0 + h - 1.8, name, color=RED, fontsize=9,
                 fontweight="bold", zorder=6)
        ax2.text(x0 + 0.5, y0 + h - 3.4, cnt, color=RED, fontsize=8, zorder=6)
    ax2.set_xticks([]); ax2.set_yticks([])
    for s_ in ax2.spines.values():
        s_.set_edgecolor(LIGHT); s_.set_linewidth(1.2)
    ax.text(0.20, 3.62, "① 三个互不重叠的验证区", fontsize=13, color=NAVY,
            fontweight="bold")

    # 右：留出协议
    ax.text(5.55, 3.62, "② 整条航次留出", fontsize=13, color=NAVY, fontweight="bold")
    ax.add_patch(mp.FancyBboxPatch((5.55, 2.46), 5.50, 0.86,
                                   boxstyle="round,pad=0.05", fc=NODE["neutral"][0],
                                   ec=NODE["neutral"][1], lw=1.4, clip_on=False))
    ax.text(5.80, 3.06, "20 条船载重力航次", fontsize=12, color=NAVY, fontweight="bold")
    ax.text(5.80, 2.72, "同一批数据，按航次切分，而不是按测点随机切", fontsize=10,
            color=INK)
    ax.annotate("", xy=(7.30, 2.30), xytext=(7.30, 2.44),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color=GREY,
                                shrinkA=0, shrinkB=0))
    ax.add_patch(mp.FancyBboxPatch((5.55, 1.44), 2.55, 0.82,
                                   boxstyle="round,pad=0.05", fc=NODE["blue"][0],
                                   ec=NODE["blue"][1], lw=1.4, clip_on=False))
    ax.text(5.80, 1.98, "训练 · 17 条", fontsize=11.5, color=NAVY, fontweight="bold")
    ax.text(5.80, 1.66, "参与解算", fontsize=10, color=INK)
    ax.add_patch(mp.FancyBboxPatch((8.42, 1.44), 2.55, 0.82,
                                   boxstyle="round,pad=0.05", fc=NODE["pink"][0],
                                   ec=NODE["pink"][1], lw=1.4, clip_on=False))
    ax.text(8.66, 1.98, "验证 · 3 条", fontsize=11.5, color=RED, fontweight="bold")
    ax.text(8.66, 1.66, "整条留出、从未参与", fontsize=10, color=INK)
    ax.text(5.55, 1.02, "为什么按航次：若随机留点，同一条测线上相邻的点会同时进入训练与验证，"
                        "空间自相关会把精度算得虚高。", fontsize=10.5, color=RED,
            linespacing=1.6)
    ax.text(5.55, 0.44, "参数在三个区域上完全冻结（α、σ、R、Q 不重调），"
                        "因此区域间的差异只来自数据本身。", fontsize=10.5, color=INK,
            linespacing=1.6)
    fig.savefig(OUT / "fig_validation_design.png", bbox_inches="tight")
    plt.close(fig)
    print("fig_validation_design.png")


# ------------------------------------------------ 改善幅度
def fig_improvement():
    fig, ax = plt.subplots(figsize=(11.4, 3.4), dpi=200)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    regions = ["综合验证区\n108 103 点", "南海-琉球\n12 310 点", "菲律宾海中部\n6 459 点"]
    kf = np.array([4.173, 4.748, 1.455])
    ssv = np.array([4.674, 5.252, 1.607])
    ff = np.array([4.575, 5.152, 1.563])
    d_ssv = ssv - kf
    d_ff = ff - kf
    p_ssv = d_ssv / ssv * 100
    p_ff = d_ff / ff * 100

    x = np.arange(3); w = 0.30
    b1 = ax.bar(x - w / 2, d_ssv, w, color=NAVY, label="相对 SSV33.1 的改善", zorder=3)
    b2 = ax.bar(x + w / 2, d_ff, w, color=LIGHT, label="相对频域融合的改善", zorder=3)
    for bars, pct in ((b1, p_ssv), (b2, p_ff)):
        for b, p in zip(bars, pct):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                    f"{b.get_height():.3f}\n({p:.1f}%)", ha="center", va="bottom",
                    fontsize=10, color=INK, fontweight="bold", linespacing=1.4)
    ax.set_xticks(x); ax.set_xticklabels(regions, fontsize=10.5)
    ax.set_ylabel("RMSE 改善量 (mGal)", fontsize=10.5)
    ax.set_ylim(0, 0.66)
    ax.legend(frameon=False, fontsize=10, ncol=2, loc="upper center")
    ax.grid(axis="y", ls=":", lw=0.6, color="0.86", zorder=0); ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.savefig(OUT / "fig_improvement.png", bbox_inches="tight")
    plt.close(fig)
    print("fig_improvement.png")


if __name__ == "__main__":
    fig_validation_design()
    fig_improvement()
