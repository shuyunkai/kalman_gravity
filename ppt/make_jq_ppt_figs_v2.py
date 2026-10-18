"""参考模板风格的新图（红字强调、面板化、箭头标注）。

踩过的两个坑已固化到此:
  1) subplots_adjust(left=0,right=1,bottom=0,top=1) —— 让 1 坐标单位 = 1 英寸;
  2) 所有 patch 设 clip_on=False, 且 xlim/ylim 留出余量 —— 避免边线被裁。
"""
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


def _box(ax, x, y, w, h, kind, title=None, lines=(), ts=13, fs=10.5, pad="round,pad=0.06"):
    fill, edge, txt = NODE[kind] if isinstance(kind, str) else (kind[0], kind[1], kind[2])
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle=pad, fc=fill, ec=edge,
                                   lw=1.5, zorder=3, clip_on=False))
    if title:
        ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center",
                fontsize=ts, fontweight="bold", color=txt, zorder=4)
    for i, s in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.74 - i * 0.30, s, ha="center", va="center",
                fontsize=fs, color=txt, zorder=4)


def _arrow(ax, x1, y1, x2, y2, color=NAVY, lw=2.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                shrinkA=0, shrinkB=0), zorder=2)


# ------------------------------------------------ 1. 什么是地球重力场
def fig_gravity():
    fig, ax = _canvas(11.4, 3.9)
    # 左：剖面示意
    ax.plot([0.35, 4.55], [2.05, 2.05], color=INK, lw=2.0, zorder=2)
    ax.text(0.35, 2.22, "地表", fontsize=10.5, color=GREY)
    # 地下高密度体
    ax.add_patch(mp.Ellipse((2.92, 0.92), 1.05, 0.54, fc="#B7A9D6", ec="#8E7DB8",
                            lw=1.4, zorder=3, clip_on=False))
    ax.text(2.92, 0.92, "高密度体", ha="center", va="center", fontsize=9.5,
            color="#2E2450", zorder=4)
    # 测点
    ax.plot(1.55, 2.05, "o", ms=8, color=RED, zorder=5)
    ax.text(1.35, 2.22, "测点 P", fontsize=10.5, color=RED, fontweight="bold")
    # 两个矢量
    _arrow(ax, 1.55, 2.00, 1.55, 1.30, color=GREY, lw=2.2)
    ax.text(1.66, 1.55, "g 理论", fontsize=12, color=GREY)
    _arrow(ax, 1.55, 2.00, 1.72, 1.18, color=RED, lw=2.6)
    ax.text(1.80, 0.96, "g 实测", fontsize=12, color=RED, fontweight="bold")
    ax.text(0.35, 0.30, "按经纬度与高程可算出理论重力；\n实测值与理论值之差即重力异常 Δg",
            fontsize=11.5, color=INK, linespacing=1.6)
    ax.text(0.35, 3.62, "① 重力异常的来源", fontsize=13.5, color=NAVY, fontweight="bold")

    # 右：重力异常图
    ax2 = fig.add_axes([0.44, 0.10, 0.30, 0.74])
    ds_z = np.load(OUT / "_grav_cache.npy") if (OUT / "_grav_cache.npy").exists() else None
    if ds_z is None:
        ax2.axis("off")
    else:
        im = ax2.imshow(ds_z, cmap="Blues_r", origin="lower", interpolation="nearest",
                        aspect="auto")
        ax2.set_xticks([]); ax2.set_yticks([])
        for s in ax2.spines.values():
            s.set_edgecolor(LIGHT); s.set_linewidth(1.2)
        cb = fig.colorbar(im, ax=ax2, fraction=0.040, pad=0.02)
        cb.ax.tick_params(labelsize=8)
        cb.set_label("重力异常 (mGal)", fontsize=9)
    ax.text(5.05, 3.62, "② 它在空间上的样子", fontsize=13.5, color=NAVY,
            fontweight="bold")
    ax.text(5.05, 0.26, "要重建的就是这张场：把它从 1′ 加密到 15″",
            fontsize=11.5, color=INK)
    fig.savefig(OUT / "fig_gravity.png", bbox_inches="tight")
    plt.close(fig)
    print("fig_gravity.png")


# ------------------------------------------------ 2. 1′ 测高重力路线
def fig_altimetry():
    fig, ax = _canvas(11.4, 3.2)
    steps = [("①", "卫星测高", "测海面高\n精度厘米级", "neutral"),
             ("②", "海面坡度", "沿轨求导\n得垂线偏差", "blue"),
             ("③", "反演重力", "逆 Vening-Meinesz\n或谱方法", "purple"),
             ("④", "1′ 重力模型", "全球覆盖\n1′ 网格", "green")]
    w, h, gap = 2.35, 1.85, 0.62
    for k, (num, title, desc, kind) in enumerate(steps):
        x = 0.20 + k * (w + gap)
        _box(ax, x, 0.62, w, h, kind, None, ())
        ax.text(x + 0.24, 0.62 + h - 0.32, num, fontsize=13, fontweight="bold",
                color=NODE[kind][2], zorder=4)
        ax.text(x + 0.66, 0.62 + h - 0.32, title, fontsize=13, fontweight="bold",
                color=NODE[kind][2], zorder=4)
        ax.text(x + 0.24, 0.62 + h - 0.88, desc, fontsize=10.5,
                color=NODE[kind][2], zorder=4, linespacing=1.5)
        if k < 3:
            _arrow(ax, x + w + 0.12, 0.62 + h / 2, x + w + gap - 0.14, 0.62 + h / 2)
    ax.text(0.20, 2.90, "卫星测高反演：从海面高到 1′ 重力模型", fontsize=13.5,
            color=NAVY, fontweight="bold")
    ax.text(0.20, 0.24, "这条路线不需要船测，因此能做到全球覆盖 —— 代价是分辨率停在 1′",
            fontsize=11, color=INK)
    fig.savefig(OUT / "fig_altimetry.png", bbox_inches="tight")
    plt.close(fig)
    print("fig_altimetry.png")


# ------------------------------------------------ 3. 四个卡点
def fig_problems():
    fig, ax = _canvas(11.4, 3.0)
    items = [
        ("①", "只用一类数据", ["测高类只用测高", "插值类只用船测"]),
        ("②", "权重靠人为给定", ["截止波长人为定", "融合权重人为定"]),
        ("③", "只输出一张场图", ["无逐像元方差", "不知哪里可信"]),
        ("④", "15″ 无成熟产品", ["公开产品均为 1′", "要 15″ 得自己重建"]),
    ]
    w, gap = 2.62, 0.42
    for k, (num, title, lines) in enumerate(items):
        x = 0.16 + k * (w + gap)
        ax.add_patch(mp.FancyBboxPatch((x, 0.34), w, 2.00,
                                       boxstyle="round,pad=0.05",
                                       fc="#FBEEEC", ec="#E3A7B0", lw=1.4,
                                       zorder=3, clip_on=False))
        ax.text(x + 0.20, 2.02, num, fontsize=13.5, fontweight="bold", color=RED,
                zorder=4)
        ax.text(x + 0.48, 2.02, title, fontsize=12, fontweight="bold",
                color=RED, zorder=4)
        for j, s in enumerate(lines):
            ax.text(x + 0.20, 1.52 - j * 0.34, s, fontsize=10.5, color=INK,
                    zorder=4, va="center")
    ax.text(0.16, 2.68, "四个卡点：都不是算法不够好，而是信息没有进同一套方程",
            fontsize=13.5, color=NAVY, fontweight="bold")
    fig.savefig(OUT / "fig_problems.png", bbox_inches="tight")
    plt.close(fig)
    print("fig_problems.png")


# ------------------------------------------------ 4. 问题 -> 卡尔曼要素
def fig_mapping():
    fig, ax = _canvas(11.4, 3.4)
    rows = [("只用一类数据", "观测向量 z", "三类观测写进同一套方程"),
            ("权重靠人为给定", "观测噪声 R / 过程噪声 Q", "权重由增益按精度自动算出"),
            ("只输出一张场图", "状态协方差 P", "逐像元方差随场输出"),
            ("15″ 无成熟产品", "状态向量 x 定义在 15″", "解在 1′ 观测里被递推出来")]
    ax.text(1.33, 3.12, "卡点", fontsize=12.5, color=GREY, fontweight="bold",
            ha="center")
    ax.text(4.76, 3.12, "卡尔曼滤波里的对应要素", fontsize=12.5, color=NAVY,
            fontweight="bold", ha="center")
    ax.text(9.16, 3.12, "解决方式", fontsize=12.5, color=NAVY,
            fontweight="bold", ha="center")
    for i, (a, b, c) in enumerate(rows):
        y = 2.42 - i * 0.62
        ax.add_patch(mp.FancyBboxPatch((0.35, y - 0.22), 1.95, 0.44,
                                       boxstyle="round,pad=0.03", fc="#FBEEEC",
                                       ec="#E3A7B0", lw=1.2, zorder=3, clip_on=False))
        ax.text(1.32, y, a, ha="center", va="center", fontsize=11, color=RED, zorder=4)
        _arrow(ax, 2.42, y, 2.90, y, color=GREY, lw=1.6)
        ax.add_patch(mp.FancyBboxPatch((2.98, y - 0.22), 3.55, 0.44,
                                       boxstyle="round,pad=0.03", fc=NODE["yellow"][0],
                                       ec=NODE["yellow"][1], lw=1.2, zorder=3, clip_on=False))
        ax.text(4.76, y, b, ha="center", va="center", fontsize=11,
                color=NODE["yellow"][2], zorder=4)
        _arrow(ax, 6.62, y, 7.10, y, color=GREY, lw=1.6)
        ax.add_patch(mp.FancyBboxPatch((7.18, y - 0.22), 3.95, 0.44,
                                       boxstyle="round,pad=0.03", fc=NODE["green"][0],
                                       ec=NODE["green"][1], lw=1.2, zorder=3, clip_on=False))
        ax.text(9.16, y, c, ha="center", va="center", fontsize=11,
                color=NODE["green"][2], zorder=4)
    fig.savefig(OUT / "fig_mapping.png", bbox_inches="tight")
    plt.close(fig)
    print("fig_mapping.png")


# ------------------------------------------------ 5. 本项目的卡尔曼流程
def fig_project_flow():
    fig, ax = _canvas(11.4, 4.2)
    ax.text(0.20, 3.95, "状态 3840 × 4800 = $1.84\\times10^{7}$ 个未知数；"
                        "卫星观测仅 960 × 1200 = $1.15\\times10^{6}$ 个",
            fontsize=11.5, color=INK, zorder=20)
    stages = [
        ("先验", "SSV 块填充\n+ α×地形高频", "neutral"),
        ("预测", "高斯平滑外推\nP 随之增大", "blue"),
        ("卫星校正", "1′ 块均值\nR_sat = 9", "purple"),
        ("船测校正", "逐像元更新\nR_ship = 1", "yellow"),
        ("RTS 平滑", "反向递推\n抑制路径效应", "green"),
        ("输出", "15″ 场\n+ 逐像元方差", "blue"),
    ]
    w, h, gap = 1.60, 2.15, 0.28
    for k, (title, desc, kind) in enumerate(stages):
        x = 0.20 + k * (w + gap)
        _box(ax, x, 1.28, w, h, kind, None, ())
        ax.text(x + w / 2, 1.28 + h - 0.34, title, ha="center", va="center",
                fontsize=12, fontweight="bold", color=NODE[kind][2], zorder=4)
        ax.text(x + w / 2, 1.28 + h - 1.05, desc, ha="center", va="center",
                fontsize=10, color=NODE[kind][2], zorder=4, linespacing=1.6)
        if k < 5:
            _arrow(ax, x + w + 0.04, 1.28 + h / 2, x + w + gap - 0.04, 1.28 + h / 2,
                   lw=1.8)
    # 递推回环
    ax.plot([9.72, 9.72, 1.00], [1.20, 0.72, 0.72], color=RED, lw=1.6, zorder=2)
    ax.annotate("", xy=(1.00, 1.22), xytext=(1.00, 0.72),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color=RED,
                                shrinkA=0, shrinkB=0))
    ax.text(5.30, 0.52, "六轮递推（预测—校正），再反向平滑一次", ha="center",
            fontsize=10.5, color=RED)
    ax.text(0.20, 0.22, "参数全域冻结：α = 0.5，σ = 4 像元，R_sat = 9，R_ship = 1，Q = 4 mGal²",
            fontsize=10.5, color=GREY)
    fig.savefig(OUT / "fig_project_flow.png", bbox_inches="tight")
    plt.close(fig)
    print("fig_project_flow.png")


if __name__ == "__main__":
    fig_gravity()
    fig_altimetry()
    fig_problems()
    fig_mapping()
    fig_project_flow()
