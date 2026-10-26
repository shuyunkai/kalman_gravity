"""干净版 KF 流程图：输入/预处理 → KF 循环 → 输出。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = r"output_real/three_parts/newdeck"
NAVY, RED, GOLD, LIGHT = "#0B2545", "#C0392B", "#C9A227", "#EEF3F9"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})

fig, ax = plt.subplots(figsize=(13.4, 5.2), dpi=220)
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)


def box(x, y, w, h, title, sub="", fc="white", fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.010,rounding_size=0.02",
                                fc=fc, ec=NAVY, lw=1.1))
    ax.text(x+w/2, y+h*0.66, title, ha="center", va="center", fontsize=fs,
            color=NAVY, fontweight="bold")
    if sub:
        ax.text(x+w/2, y+h*0.29, sub, ha="center", va="center", fontsize=fs-1.5,
                color="#5B6573")


def arr(p0, p1, rad=0.0, color="#5B6573"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=1.3,
                                 connectionstyle=f"arc3,rad={rad}"))


ax.text(0.13, 0.95, "输入与预处理", ha="center", fontsize=12, color=GOLD,
        fontweight="bold")
ax.text(0.58, 0.95, "迭代 KF（6 轮）", ha="center", fontsize=12, color=GOLD,
        fontweight="bold")
ax.text(0.90, 0.95, "输出", ha="center", fontsize=12, color=GOLD,
        fontweight="bold")

box(0.02, 0.76, 0.20, 0.13, "卫星 1′", "SSV33.1 / DTU21")
box(0.02, 0.57, 0.20, 0.13, "船测重力", "JODC / 航次")
box(0.02, 0.38, 0.20, 0.13, "地形 GEBCO", "15″ 水深")
box(0.25, 0.50, 0.16, 0.18, "块平均 + 先验", "1′ 块观测\nSSV + 地形高频", fc="#FFF8E7")

for y in (0.825, 0.635, 0.445):
    arr((0.22, y), (0.25, 0.59))

box(0.46, 0.72, 0.23, 0.13, "① 预测", "$x^-$、$P^-$", fc=LIGHT)
box(0.73, 0.72, 0.23, 0.13, "② 卫星更新", "块观测 H\n标量增益", fc=LIGHT)
box(0.73, 0.48, 0.23, 0.13, "③ 船测更新", "点更新 + 邻域", fc=LIGHT)
box(0.46, 0.48, 0.23, 0.13, "④ 6 轮迭代", "修正量收敛", fc=LIGHT)
arr((0.69, 0.785), (0.73, 0.785))
arr((0.845, 0.72), (0.845, 0.61))
arr((0.73, 0.545), (0.69, 0.545))
arr((0.575, 0.61), (0.575, 0.72), rad=0.0)
ax.text(0.585, 0.665, "下一轮", fontsize=7, color=RED)
ax.text(0.855, 0.665, "R/Q", fontsize=7, color=RED)

box(0.46, 0.26, 0.36, 0.13, "RTS 反向平滑", "输出 $x_{smooth}$、$P_{smooth}$", fc="#FFF8E7")
arr((0.575, 0.48), (0.575, 0.39))

box(0.86, 0.32, 0.12, 0.12, "15″ 场", "1.85 km")
box(0.86, 0.15, 0.12, 0.12, "逐像元方差", "$\\sqrt{P}$")
arr((0.82, 0.37), (0.86, 0.38))
arr((0.82, 0.29), (0.86, 0.21))

ax.text(0.5, 0.02,
        "先验：$x_0$ = SSV 块平均 + α·地形布格高频；$P_0$、$Q$、$R$ 固定；"
        "相干保持预测：$x^-=(1-wC^2)G_\\sigma x+wC^2x$",
        ha="center", fontsize=8, color=RED)

fig.savefig(f"{ROOT}/fig_new_kf_flow_full.png", bbox_inches="tight")
print("saved")
