"""本项目 KF 流程：干净、连线完整版。"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, RED, GOLD, LIGHT = "#0B2545", "#C0392B", "#C9A227", "#EEF3F9"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 8})

fig, ax = plt.subplots(figsize=(13.4, 4.6), dpi=240)
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)


def box(x, y, w, h, title, sub="", fc="white", fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.008,rounding_size=0.018",
                                fc=fc, ec=NAVY, lw=1.0))
    ax.text(x+w/2, y+h*0.64, title, ha="center", va="center",
            fontsize=fs, color=NAVY, fontweight="bold")
    if sub:
        ax.text(x+w/2, y+h*0.28, sub, ha="center", va="center",
                fontsize=fs-1.4, color="#5B6573")


def arr(p0, p1, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12,
                                 color="#5B6573", lw=1.2,
                                 connectionstyle=f"arc3,rad={rad}"))


ax.text(0.09, 0.95, "输入", ha="center", fontsize=10, color=GOLD, fontweight="bold")
ax.text(0.27, 0.95, "预处理", ha="center", fontsize=10, color=GOLD, fontweight="bold")
ax.text(0.55, 0.95, "迭代 KF（6 轮）", ha="center", fontsize=10, color=GOLD, fontweight="bold")
ax.text(0.87, 0.95, "输出", ha="center", fontsize=10, color=GOLD, fontweight="bold")

box(0.02, 0.72, 0.15, 0.13, "卫星 1′", "SSV33.1", fs=8.5)
box(0.02, 0.49, 0.15, 0.13, "船测重力", "JODC / 航次", fs=8.5)
box(0.02, 0.26, 0.15, 0.13, "地形 GEBCO", "15″ 水深", fs=8.5)
box(0.20, 0.43, 0.15, 0.22, "块平均 + 先验", "1′ 块观测\nSSV + 地形高频", fc="#FFF8E7", fs=8.2)
for y in (0.785, 0.555, 0.325):
    arr((0.17, y), (0.20, 0.54))

box(0.39, 0.60, 0.15, 0.14, "① 预测", "$x^-$、$P^-$", fc=LIGHT, fs=8.5)
box(0.57, 0.60, 0.15, 0.14, "② 卫星更新", "块观测 H\n增益", fc=LIGHT, fs=8.5)
box(0.57, 0.38, 0.15, 0.14, "③ 船测更新", "点更新 + 邻域", fc=LIGHT, fs=8.5)
box(0.39, 0.38, 0.15, 0.14, "④ 6 轮迭代", "修正量收敛", fc=LIGHT, fs=8.5)
arr((0.35, 0.66), (0.39, 0.67))
arr((0.54, 0.67), (0.57, 0.67))
arr((0.645, 0.60), (0.645, 0.52))
arr((0.57, 0.45), (0.54, 0.45))
arr((0.465, 0.52), (0.465, 0.60))
ax.text(0.475, 0.56, "下一轮", fontsize=7, color=RED)

box(0.39, 0.14, 0.33, 0.13, "RTS 反向平滑", "输出 x_smooth、P_smooth", fc="#FFF8E7", fs=8.5)
arr((0.465, 0.38), (0.465, 0.27))
ax.text(0.482, 0.32, "6 轮后", fontsize=7, color=RED)

box(0.79, 0.32, 0.18, 0.14, "15″ 重力场", "0.46 km 格距", fc=LIGHT, fs=8.5)
box(0.79, 0.11, 0.18, 0.13, "逐像元方差", "$\\sqrt{P}$", fc=LIGHT, fs=8.5)
arr((0.72, 0.22), (0.79, 0.39))
arr((0.72, 0.19), (0.79, 0.17))

ax.text(0.5, 0.02,
        "先验：$x_0$ = SSV33.1 块平均 + α·地形高频（α 为先验权重）；$P_0$、Q、R 固定；相干预测：A=(1−wC²)Gσ+wC²I，$x^-=Ax$",
        ha="center", fontsize=7.2, color=RED)
fig.savefig(OUT / "fig_new_kf_flow_full.png", bbox_inches="tight")
print("saved")
