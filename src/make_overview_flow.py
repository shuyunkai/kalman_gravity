"""总体流程：五步，简洁版。"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, GOLD, RED = "#0B2545", "#C9A227", "#C0392B"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False})

fig, ax = plt.subplots(figsize=(13.4, 4.0), dpi=240)
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
steps = [
    ("01", "多源数据统一", "4 套 1′ + 船测 + 地形\n统一到 15″"),
    ("02", "块观测建模", "1′ 块平均\nH、R/Q"),
    ("03", "相干结构估计", "梯度 → 张量\n→ 相干 C"),
    ("04", "自适应预测", "A=(1−wC²)Gσ+wC²I\n脊线少平滑"),
    ("05", "迭代解算与验证", "6 轮 + RTS\n三区留出航次"),
]
n = len(steps); w = 0.165; gap = 0.035
x0 = 0.035
for i, (num, head, sub) in enumerate(steps):
    x = x0 + i*(w+gap)
    ax.add_patch(FancyBboxPatch((x, 0.28), w, 0.44,
                                boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc="white", ec="#B9C6D6", lw=1.1))
    ax.add_patch(plt.Circle((x+0.028, 0.66), 0.020, color=NAVY, zorder=5))
    ax.text(x+0.028, 0.66, num, color=GOLD, fontsize=6.5, ha="center", va="center", zorder=6)
    ax.text(x+0.060, 0.66, head, color=NAVY, fontsize=8.6,
            fontweight="bold", ha="left", va="center")
    ax.text(x+w/2, 0.47, sub, color="#333333", fontsize=7.5,
            ha="center", va="center", linespacing=1.5)
    if i < n-1:
        ax.add_patch(FancyArrowPatch((x+w+0.004, 0.50), (x+w+gap-0.004, 0.50),
                                     arrowstyle="-|>", mutation_scale=13,
                                     color=GOLD, lw=1.6))
ax.text(0.5, 0.12, "后续逐段展开：每一步的算子和对应结果图", ha="center",
        fontsize=8, color="#44546A")
fig.savefig(OUT / "fig_overview_flow.png", bbox_inches="tight")
print("saved")
