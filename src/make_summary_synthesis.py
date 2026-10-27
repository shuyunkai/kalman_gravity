"""总结页：输入 → 方法 → 输出，只讲做了什么与为什么。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = r"output_real/three_parts/newdeck"
NAVY, GOLD, RED, PALE = "#0B2545", "#C9A227", "#C0392B", "#EEF3F9"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})

fig, ax = plt.subplots(figsize=(13.6, 4.2), dpi=220)
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

steps = [
    (0.03, "输入", "1′ + 船测 + 地形", "补 5–20 km 短波"),
    (0.36, "方法", "相干保持 KF", "C 改写预测矩阵，脊线少平滑、平地正常平滑"),
    (0.69, "输出", "15″ 场 + $P$", "同一套场看精度与断裂"),
]
for x, tag, main, why in steps:
    ax.text(x+0.14, 0.86, tag, ha="center", fontsize=10, color=GOLD,
            fontweight="bold")
    ax.add_patch(FancyBboxPatch((x, 0.52), 0.28, 0.26,
                                boxstyle="round,pad=0.012,rounding_size=0.03",
                                fc="#F7F9FC", ec=NAVY, lw=1.2))
    ax.text(x+0.14, 0.65, main, ha="center", va="center", fontsize=12,
            color=NAVY, fontweight="bold")
    ax.text(x+0.14, 0.40, why, ha="center", va="center", fontsize=8.2,
            color="#44546A", wrap=True)

for x in (0.315, 0.645):
    ax.add_patch(FancyArrowPatch((x, 0.65), (x+0.035, 0.65),
                                 arrowstyle="-|>", mutation_scale=16,
                                 color=RED, lw=1.5))

ax.add_patch(FancyBboxPatch((0.03, 0.10), 0.94, 0.12,
                            boxstyle="round,pad=0.012,rounding_size=0.03",
                            fc="#FFF8E7", ec=GOLD, lw=1))
ax.text(0.5, 0.16,
        "不另做一套场：避免重力精度与断裂结构“两张皮”",
        ha="center", va="center", fontsize=10, color=RED, fontweight="bold")

fig.savefig(f"{ROOT}/fig_new_summary_synthesis.png", bbox_inches="tight")
print("saved")
