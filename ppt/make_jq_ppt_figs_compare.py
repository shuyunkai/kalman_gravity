"""新增对照图：六类做法在四个关键维度上的能力矩阵。

用法: python make_jq_ppt_figs_compare.py  （或由 make_jq_ppt_figs 调用 fig_compare）
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "savefig.facecolor": "white",
})
NAVY, LIGHT, GREY, RED = "#16365C", "#D8E0EC", "#5B6573", "#C0392B"
PALE, MID = "#EEF3FA", "#4A7AA7"
OUT = Path(r"D:\gravity\ppt_jq\figs")

COLS = ["多源同时约束", "权重由数据定", "输出不确定度", "恢复亚 1′ 短波"]
ROWS = [
    ("测高重力模型",           [0, 2, 0, 0]),
    ("空间插值",               [0, 0, 0, 0]),
    ("频域分频融合",           [1, 0, 0, 1]),
    ("测高与船测协同融合",     [1, 0, 0, 1]),
    ("深度学习超分",           [1, 1, 0, 1]),
    ("卡尔曼滤波（本方法）",   [2, 2, 2, 2]),
]
# 0 = 不具备, 1 = 部分具备, 2 = 具备
STYLE = {0: (LIGHT, "×"), 1: (MID, "○"), 2: (NAVY, "●")}


def fig_compare():
    nr, nc = len(ROWS), len(COLS)
    fig, ax = plt.subplots(figsize=(10.6, 4.5), dpi=200)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 10.6); ax.set_ylim(0, 4.5); ax.axis("off")

    x0, y0 = 3.30, 0.62          # 矩阵左上角
    cw, ch = 1.62, 0.56
    for j, c in enumerate(COLS):
        ax.text(x0 + cw * (j + 0.5), y0 + ch * nr + 0.26, c, ha="center",
                va="center", fontsize=12, fontweight="bold", color=NAVY)

    for i, (name, vals) in enumerate(ROWS):
        y = y0 + ch * (nr - 1 - i)
        last = (i == nr - 1)
        if last:
            ax.add_patch(mp.Rectangle((0.10, y - 0.03), 10.35, ch + 0.06,
                                      fc=PALE, ec="none", zorder=0))
        elif i % 2 == 1:
            ax.add_patch(mp.Rectangle((0.10, y - 0.03), 10.35, ch + 0.06,
                                      fc="#FAFBFC", ec="none", zorder=0))
        ax.text(0.28, y + ch / 2, name, ha="left", va="center", fontsize=12,
                fontweight="bold" if last else "normal",
                color=NAVY if last else "#1C2430")
        for j, v in enumerate(vals):
            cx = x0 + cw * (j + 0.5)
            color, mark = STYLE[v]
            ax.text(cx, y + ch / 2, mark, ha="center", va="center",
                    fontsize=17 if v == 2 else 15, color=color,
                    fontweight="bold")

    ax.text(0.28, 0.18, "● 具备    ○ 部分具备    × 不具备"
                        "　　每一项都对应前页方程里的一个要素",
            fontsize=10.5, color=GREY)
    fig.savefig(OUT / "fig_compare.png", bbox_inches="tight")
    plt.close(fig)
    print("fig_compare.png 已生成")


if __name__ == "__main__":
    fig_compare()
