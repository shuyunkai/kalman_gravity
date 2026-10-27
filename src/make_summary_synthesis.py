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
         ("5.0–8.6%", "精度改善（三区）", "相对最优常用 15″ 场"),
         ("81.0%", "5 km 断裂覆盖", "命中 24.0%")]
for i, (val, name, sub) in enumerate(cards):
    ax = fig.add_subplot(gs[0, i]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0.03, 0.05), 0.94, 0.90,
                                boxstyle="round,pad=0.015,rounding_size=0.03",
                                fc="#EEF3F9", ec=NAVY, lw=1.2))
    ax.text(0.5, 0.66, val, ha="center", fontsize=22, color=RED, fontweight="bold")
    ax.text(0.5, 0.35, name, ha="center", fontsize=11, color=NAVY, fontweight="bold")
    ax.text(0.5, 0.15, sub, ha="center", fontsize=8, color="#5B6573")

ax = fig.add_subplot(gs[1, 0])
names = ["南海-琉球", "九州东侧", "日本"]
vals = [8.6, 5.0, 5.2]
ax.barh(names, vals, color=[NAVY, NAVY, NAVY], height=0.55)
for i, v in enumerate(vals):
    ax.text(v + 0.15, i, f"{v}%", va="center", fontsize=8)
ax.set_xlim(0, 12); ax.set_xlabel("相对最优常用 15″ 场改善 (%)", fontsize=8)
ax.set_title("(a) 相对最优常用 15″ 场的改善", fontsize=10)
ax.tick_params(labelsize=8); ax.grid(axis="x", alpha=0.2)

ax = fig.add_subplot(gs[1, 1])
labels = ["5 km 覆盖", "5 km 命中"]
vals2 = [81.0, 24.0]
ax.barh(labels, vals2, color=[NAVY, GOLD], height=0.5)
for i, v in enumerate(vals2):
    ax.text(v+1, i, f"{v}%", va="center", fontsize=8)
ax.set_xlim(0, 100); ax.set_xlabel("百分比 (%)", fontsize=8)
ax.set_title("(b) 断裂识别指标", fontsize=10)
ax.tick_params(labelsize=8); ax.grid(axis="x", alpha=0.2)

ax = fig.add_subplot(gs[1, 2])
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("(c) 方法链", fontsize=10)
steps = ["1′ + 船测 + 地形", "相干保持 KF", "15″ 场 + P", "精度 / 断裂"]
for i, s in enumerate(steps):
    y = 0.78 - i*0.245
    ax.add_patch(FancyBboxPatch((0.05, y), 0.90, 0.16,
                                boxstyle="round,pad=0.012,rounding_size=0.03",
                                fc="#EEF3F9" if i < 3 else "#FFF8E7",
                                ec=NAVY if i < 3 else GOLD, lw=1))
    ax.text(0.5, y+0.08, s, ha="center", va="center", fontsize=9,
            color=NAVY if i < 3 else RED, fontweight="bold" if i == 3 else "normal")
    if i < 3:
        ax.add_patch(FancyArrowPatch((0.5, y-0.005), (0.5, y-0.045),
                                     arrowstyle="-|>", mutation_scale=12,
                                     color="#78909C", lw=1.2))

fig.text(0.5, 0.015,
         "同一套 15″ 场同时给出重力精度与断裂指标，不再为两个任务各做一套场",
         ha="center", fontsize=9, color=NAVY)
fig.savefig(f"{ROOT}/fig_new_summary_synthesis.png", bbox_inches="tight")
print("saved")
