"""总结页：五步流程 + 每步依据（不出现“做了什么/为什么”字样）。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = r"output_real/three_parts/newdeck"
NAVY, GOLD, RED, PALE = "#0B2545", "#C9A227", "#C0392B", "#EEF3F9"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})

fig, ax = plt.subplots(figsize=(13.6, 4.5), dpi=220)
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

rows = [
    ("01", "输入", "1′ + 船测 + 地形", "补 5–20 km 短波"),
    ("02", "先验", "SSV 块平均 + 地形高频", "给预测一个初始结构"),
    ("03", "预测", "$A=(1-wC^2)G_\\sigma+wC^2I$", "脊线少平滑，平地正常平滑"),
    ("04", "更新", "块观测 + 船测邻域，6 轮 + RTS", "观测逐轮修正，不只改单点"),
    ("05", "结果", "15″ / 1.85 km；较最优常用场改善 5.0–8.6%", "断裂覆盖 81.0%，命中 24.0%"),
    ("06", "应用", "15″ 高精度重力场", "导航、地形/工程、资源与断裂识别"),
]
for i, (num, tag, action, reason) in enumerate(rows):
    y = 0.80 - i*0.145
    ax.add_patch(FancyBboxPatch((0.02, y), 0.16, 0.125,
                                boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc=NAVY, ec=NAVY))
    ax.text(0.045, y+0.082, num, fontsize=7, color=GOLD, fontweight="bold",
            va="center")
    ax.text(0.045, y+0.043, tag, fontsize=9, color="white", va="center")

    ax.add_patch(FancyBboxPatch((0.20, y), 0.39, 0.125,
                                boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc="white", ec="#C9D3DF", lw=0.9))
    ax.text(0.225, y+0.062, action, fontsize=8.8,
            color=RED if i == 4 else NAVY, va="center")

    ax.add_patch(FancyBboxPatch((0.61, y), 0.37, 0.125,
                                boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc="#FFF8E7", ec="#E8D9A8", lw=0.9))
    ax.text(0.635, y+0.062, reason, fontsize=8.2,
            color=NAVY if i >= 4 else "#44546A", va="center")

    if i < 5:
        ax.add_patch(FancyArrowPatch((0.10, y-0.006), (0.10, y-0.031),
                                     arrowstyle="-|>", mutation_scale=9,
                                     color="#B0BEC5", lw=1))

fig.savefig(f"{ROOT}/fig_new_summary_synthesis.png", bbox_inches="tight")
print("saved")
