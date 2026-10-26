"""总结页：KPI + 三区改善 + 断裂指标 + 精度—结构权衡。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = r"output_real/three_parts/newdeck"
NAVY, GOLD, RED, PALE = "#0B2545", "#C9A227", "#C0392B", "#D6E2F0"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})

fig = plt.figure(figsize=(13.4, 5.4), dpi=220)
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.65], hspace=0.42, wspace=0.32)

cards = [("15″", "分辨率", "1.85 km 网格"),
         ("0.2–8.7%", "三区精度改善", "相对 1′ 产品"),
         ("67.9%", "5 km 断裂覆盖", "命中 44.5%")]
for i, (val, name, sub) in enumerate(cards):
    ax = fig.add_subplot(gs[0, i]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0.03, 0.05), 0.94, 0.90,
                                boxstyle="round,pad=0.015,rounding_size=0.03",
                                fc="#EEF3F9", ec=NAVY, lw=1.2))
    ax.text(0.5, 0.66, val, ha="center", fontsize=22, color=RED, fontweight="bold")
    ax.text(0.5, 0.35, name, ha="center", fontsize=11, color=NAVY, fontweight="bold")
    ax.text(0.5, 0.15, sub, ha="center", fontsize=8, color="#5B6573")

ax = fig.add_subplot(gs[1, 0])
names = ["南海-琉球", "菲律宾海", "日本"]
vals = [8.7, 0.2, 6.3]
ax.barh(names, vals, color=[NAVY, "#90A4AE", NAVY], height=0.55)
for i, v in enumerate(vals):
    ax.text(v+0.2, i, f"{v}%", va="center", fontsize=8)
ax.set_xlim(0, 10); ax.set_xlabel("RMSE 改善 (%)", fontsize=8)
ax.set_title("(a) 三区精度改善", fontsize=10)
ax.tick_params(labelsize=8); ax.grid(axis="x", alpha=0.2)

ax = fig.add_subplot(gs[1, 1])
labels = ["5 km 覆盖", "5 km 命中"]
vals2 = [67.9, 44.5]
ax.barh(labels, vals2, color=[NAVY, GOLD], height=0.5)
for i, v in enumerate(vals2):
    ax.text(v+1, i, f"{v}%", va="center", fontsize=8)
ax.set_xlim(0, 100); ax.set_xlabel("百分比 (%)", fontsize=8)
ax.set_title("(b) 断裂识别指标", fontsize=10)
ax.tick_params(labelsize=8); ax.grid(axis="x", alpha=0.2)

ax = fig.add_subplot(gs[1, 2])
ax.scatter([3.88], [62.1], s=70, c="#2E5C8A", zorder=3)
ax.scatter([3.71], [67.9], s=90, c=RED, zorder=3)
ax.annotate("", xy=(3.73, 67.3), xytext=(3.86, 62.7),
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4))
ax.text(3.885, 61.4, "标准 KF", fontsize=8, color="#2E5C8A")
ax.text(3.66, 68.5, "相干保持", fontsize=8, color=RED)
ax.set_xlim(3.62, 3.95); ax.set_ylim(58, 71)
ax.set_xlabel("留出 RMSE (mGal)", fontsize=8)
ax.set_ylabel("5 km 覆盖 (%)", fontsize=8)
ax.set_title("(c) 精度—结构权衡", fontsize=10)
ax.grid(alpha=0.2); ax.tick_params(labelsize=7)

fig.text(0.5, 0.015,
         "方法链：1′ + 船测 + 地形  →  相干保持 KF  →  15″ 场 + P  →  精度 / 断裂验证",
         ha="center", fontsize=9, color=NAVY)
fig.savefig(f"{ROOT}/fig_new_summary_synthesis.png", bbox_inches="tight")
print("saved")
