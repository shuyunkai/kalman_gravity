"""2×2 示意图：四种方法、标准 KF 四步。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = r"output_real/three_parts/newdeck"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 11})
NAVY, GREY, RED, LIGHT = "#0B2545", "#5B6573", "#C0392B", "#EEF3F9"


def card(ax, x, y, w, h, title, lines, color=LIGHT, tc=NAVY):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.015,rounding_size=0.03",
                                fc=color, ec=NAVY, lw=1.2))
    ax.text(x + w/2, y + h*0.78, title, ha="center", va="center",
            fontsize=16, fontweight="bold", color=tc)
    for i, line in enumerate(lines):
        ax.text(x + w/2, y + h*(0.52 - i*0.16), line, ha="center", va="center",
                fontsize=12, color=GREY)


def methods():
    fig, ax = plt.subplots(figsize=(13.4, 5.2), dpi=220)
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    data = [
        ("插值", ["双线性 / 克里金", "把 1′ 直接加密"], "平滑短波，短波是假的"),
        ("频域融合", ["按波长融合", "测高 + 船测"], "需要先验，边界效应"),
        ("深度学习", ["成对样本", "学习重力→地形"], "缺物理约束，样本少"),
        ("卡尔曼滤波", ["状态方程 + 观测", "多源递推"], "协方差难建模"),
    ]
    for i, (title, lines, lim) in enumerate(data):
        x = 0.03 + (i % 2) * 0.50
        y = 0.53 - (i // 2) * 0.47
        card(ax, x, y, 0.44, 0.40, title, lines,
             color="#E8D5C4" if i == 3 else LIGHT)
        ax.text(x + 0.22, y + 0.08, lim, ha="center", fontsize=11, color=RED)
    ax.text(0.5, 0.97, "四条常用路线", ha="center", fontsize=14, color=NAVY,
            fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{ROOT}/fig_new_methods_2x2.png", bbox_inches="tight")
    print("methods")


def stdkf():
    fig, ax = plt.subplots(figsize=(13.4, 5.2), dpi=220)
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    boxes = [
        (0.03, 0.55, "预测", ["$x^{-}=Ax$", "$P^{-}=APA^{T}+Q$"]),
        (0.53, 0.55, "观测", ["$z=Hx+v$", "$R$：观测噪声"]),
        (0.53, 0.08, "增益", ["$K=P^{-}H^{T}(HP^{-}H^{T}+R)^{-1}$"]),
        (0.03, 0.08, "更新", ["$x=x^{-}+K(z-Hx^{-})$", "$P=(I-KH)P^{-}$"]),
    ]
    for x, y, title, lines in boxes:
        card(ax, x, y, 0.44, 0.36, title, lines)
    ax.annotate("", xy=(0.53, 0.73), xytext=(0.47, 0.73),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8))
    ax.annotate("", xy=(0.75, 0.44), xytext=(0.75, 0.55),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8))
    ax.annotate("", xy=(0.47, 0.26), xytext=(0.53, 0.26),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8))
    ax.annotate("", xy=(0.25, 0.55), xytext=(0.25, 0.44),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8))
    ax.text(0.5, 0.97, "标准卡尔曼滤波：预测—观测—增益—更新", ha="center",
            fontsize=14, color=NAVY, fontweight="bold")
    ax.text(0.5, 0.015, "权重由 R/Q 自动分配，并给出逐像元方差",
            ha="center", fontsize=11, color=RED)
    fig.tight_layout()
    fig.savefig(f"{ROOT}/fig_new_stdkf_2x2.png", bbox_inches="tight")
    print("stdkf")


if __name__ == "__main__":
    methods()
    stdkf()
