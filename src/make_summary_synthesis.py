"""总结页：问题 → 方法 → 结果 三栏综合。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = r"output_real/three_parts/newdeck"
NAVY, GOLD, RED, PALE = "#0B2545", "#C9A227", "#C0392B", "#EEF3F9"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})

fig, ax = plt.subplots(figsize=(13.6, 4.4), dpi=220)
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

cols = [
    (0.02, "要解决的问题", "#FCEDEA", RED),
    (0.345, "怎么做", PALE, NAVY),
    (0.67, "做到什么", "#FFF8E7", GOLD),
]
for x, title, bg, accent in cols:
    ax.add_patch(FancyBboxPatch((x, 0.12), 0.30, 0.80,
                                boxstyle="round,pad=0.012,rounding_size=0.03",
                                fc=bg, ec="#C9D3DF", lw=1))
    ax.add_patch(FancyBboxPatch((x+0.02, 0.82), 0.26, 0.08,
                                boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc=NAVY, ec=NAVY))
    ax.text(x+0.15, 0.86, title, ha="center", va="center",
            fontsize=11, color=GOLD, fontweight="bold")

# 左：问题
left = [
    "1′ 产品 20 km 以下能量弱",
    "插值只平滑，频域有边界振铃",
    "LSC 对协方差函数敏感",
    "标准 KF 对角 P、单点更新",
]
for i, s in enumerate(left):
    y = 0.70 - i*0.145
    ax.text(0.045, y, "•", fontsize=12, color=RED, va="center")
    ax.text(0.065, y, s, fontsize=8.3, va="center", color="#1A1A1A", wrap=True)

# 中：方法
mid = [
    ("1", "1′ + 船测 + 地形"),
    ("2", "结构张量相干 $C$"),
    ("3", "$A=(1-wC^2)G_\\sigma+wC^2I$"),
    ("4", "块观测 + 6 轮 + RTS"),
    ("5", "15″ 场 + $P$"),
]
for i, (n, s) in enumerate(mid):
    y = 0.72 - i*0.135
    ax.add_patch(FancyBboxPatch((0.365, y), 0.26, 0.095,
                                boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc="white", ec="#B0BEC5", lw=1))
    ax.text(0.378, y+0.047, n, fontsize=7, color=GOLD, fontweight="bold",
            va="center")
    ax.text(0.405, y+0.047, s, fontsize=8.2, color=NAVY, va="center")
    if i < 4:
        ax.add_patch(FancyArrowPatch((0.495, y-0.005), (0.495, y-0.035),
                                     arrowstyle="-|>", mutation_scale=11,
                                     color="#78909C", lw=1.1))

# 右：结果
ax.text(0.82, 0.70, "15″", ha="center", fontsize=30, color=RED,
        fontweight="bold")
ax.text(0.82, 0.645, "1.85 km 网格", ha="center", fontsize=8, color="#5B6573")
ax.text(0.82, 0.55, "5.0–8.6%", ha="center", fontsize=21, color=RED,
        fontweight="bold")
ax.text(0.82, 0.495, "相对最优常用 15″ 场", ha="center", fontsize=7.5,
        color="#5B6573")
ax.text(0.82, 0.40, "81.0% / 24.0%", ha="center", fontsize=19, color=NAVY,
        fontweight="bold")
ax.text(0.82, 0.35, "断裂覆盖 / 命中", ha="center", fontsize=7.5,
        color="#5B6573")
ax.text(0.82, 0.26, "同一套 15″ 场", ha="center", fontsize=10, color=GOLD,
        fontweight="bold")
ax.text(0.82, 0.215, "同时看重力精度与短波结构", ha="center", fontsize=7.5,
        color="#5B6573")

ax.text(0.045, 0.16, "→ 所以需要 15″ 重建", fontsize=9, color=RED,
        fontweight="bold")

# 底部
ax.add_patch(FancyBboxPatch((0.02, 0.025), 0.95, 0.065,
                            boxstyle="round,pad=0.01,rounding_size=0.02",
                            fc="#F3F6FA", ec="#C9D3DF", lw=0.8))
ax.text(0.5, 0.058,
        "下一步：C / w 自适应、P 的 $APA^T$ 空间传播、提高断层命中率",
        ha="center", fontsize=8.2, color=NAVY)

fig.savefig(f"{ROOT}/fig_new_summary_synthesis.png", bbox_inches="tight")
print("saved")
