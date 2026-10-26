"""总结页：KPI + 方法链，不复用场图。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = r"output_real/three_parts/newdeck"
NAVY, GOLD, RED = "#0B2545", "#C9A227", "#C0392B"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 10})

fig = plt.figure(figsize=(13.4, 5.2), dpi=220)
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.35], hspace=0.30)

ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
cards = [("15″", "分辨率", "1.85 km 网格"),
         ("0.2–8.7%", "精度改善", "相对 1′ 产品"),
         ("67.9%", "断裂覆盖", "5 km，GM9")]
for i, (val, name, sub) in enumerate(cards):
    x = 0.02 + i*0.33
    ax.add_patch(FancyBboxPatch((x, 0.08), 0.30, 0.82,
                                boxstyle="round,pad=0.015,rounding_size=0.03",
                                fc="#EEF3F9", ec=NAVY, lw=1.2))
    ax.text(x+0.15, 0.62, val, ha="center", fontsize=24, color=RED,
            fontweight="bold")
    ax.text(x+0.15, 0.34, name, ha="center", fontsize=12, color=NAVY,
            fontweight="bold")
    ax.text(x+0.15, 0.16, sub, ha="center", fontsize=9, color="#5B6573")

ax = fig.add_subplot(gs[1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
nodes = [("1′ + 船测\n+ 地形", 0.02), ("相干保持\nKF", 0.24),
         ("15″ 重力场\n+ P", 0.46), ("精度 / 断裂\n验证", 0.68)]
for txt, x in nodes:
    ax.add_patch(FancyBboxPatch((x, 0.28), 0.16, 0.38,
                                boxstyle="round,pad=0.015,rounding_size=0.03",
                                fc="white", ec=NAVY, lw=1.2))
    ax.text(x+0.08, 0.47, txt, ha="center", va="center", fontsize=10)
for x in (0.18, 0.40, 0.62):
    ax.add_patch(FancyArrowPatch((x, 0.47), (x+0.06, 0.47),
                                 arrowstyle="-|>", mutation_scale=15,
                                 color=GOLD, lw=1.8))
ax.text(0.5, 0.90, "方法链：数据 → 相干保持 KF → 15″ 场 → 精度与断裂",
        ha="center", fontsize=12, color=NAVY, fontweight="bold")
ax.text(0.5, 0.10, "精度来自留出航次；断裂来自 5–20 km 带通 + THG",
        ha="center", fontsize=9, color=RED)
fig.savefig(f"{ROOT}/fig_new_summary_synthesis.png", bbox_inches="tight")
print("saved")
