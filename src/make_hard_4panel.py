"""第5页四困难：观测尺度、频带缺失、误差/不适定、边界/尺度。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, RED, GOLD, BLUE, GRAY = "#0B2545", "#C0392B", "#C9A227", "#7FA8C9", "#90A4AE"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})

fig, axes = plt.subplots(2, 2, figsize=(13.4, 5.2), dpi=220,
                         gridspec_kw={"wspace": 0.16, "hspace": 0.45})

# (a) observation scale
ax = axes[0, 0]; ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("(a) 观测尺度不匹配", fontsize=9)
ax.add_patch(Rectangle((0.20, 0.28), 0.42, 0.42, fill=False, ec=NAVY, lw=1.7))
for i in range(1, 4):
    ax.plot([0.20+0.105*i, 0.20+0.105*i], [0.28, 0.70], color=NAVY, lw=0.7)
    ax.plot([0.20, 0.62], [0.28+0.105*i, 0.28+0.105*i], color=NAVY, lw=0.7)
for i in range(4):
    for j in range(4):
        ax.text(0.2525+0.105*i, 0.3325+0.105*j, "?", ha="center", va="center",
                fontsize=7, color=RED)
ax.text(0.41, 0.78, "1′ 块平均 = 1 个观测", ha="center", fontsize=8, color=NAVY,
        fontweight="bold")
ax.text(0.41, 0.18, "15″ 块内变化无直接观测", ha="center", fontsize=7.5, color=RED)
ax.annotate("", xy=(0.78, 0.50), xytext=(0.65, 0.50),
            arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.2))
ax.text(0.80, 0.58, "船测/地形/先验", fontsize=7, color=GOLD)
ax.text(0.80, 0.44, "只能外部约束", fontsize=7, color=GOLD)

# (b) band gap schematic
ax = axes[0, 1]; ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("(b) 5–20 km 频带缺口", fontsize=9)
ax.add_patch(Rectangle((0.12, 0.54), 0.43, 0.14, fc=RED, alpha=0.20, ec=RED))
ax.text(0.335, 0.61, "目标 5–20 km", ha="center", va="center", fontsize=7.5, color=RED)
ax.add_patch(Rectangle((0.40, 0.30), 0.45, 0.14, fc=GRAY, alpha=0.25, ec=GRAY))
ax.text(0.625, 0.37, "1′ 产品有效 12–20 km", ha="center", va="center", fontsize=7, color=NAVY)
ax.annotate("", xy=(0.40, 0.47), xytext=(0.12, 0.47),
            arrowprops=dict(arrowstyle="<->", color=RED, lw=1.0))
ax.text(0.26, 0.75, "5–12 km 缺失", ha="center", fontsize=7, color=RED)

# (c) error amplification schematic
ax = axes[1, 0]; ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("(c) 误差进入反演、病态放大", fontsize=9)
xx = np.linspace(0, 1, 160)
ax.plot(0.06+0.26*xx, 0.50+0.025*np.sin(2*np.pi*5*xx), color=BLUE, lw=1.1)
ax.text(0.19, 0.32, "观测误差", ha="center", fontsize=7, color=BLUE)
ax.annotate("", xy=(0.55, 0.50), xytext=(0.38, 0.50),
            arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.2))
ax.text(0.465, 0.56, "反演", ha="center", fontsize=7, color=GOLD)
ax.plot(0.60+0.30*xx, 0.50+0.25*np.exp(-((xx-0.5)/0.18)**2)*np.sin(2*np.pi*9*xx),
        color=RED, lw=1.1)
ax.text(0.75, 0.18, "误差被放大", ha="center", fontsize=7.5, color=RED)

# (d) multi-scale coupling
ax = axes[1, 1]; ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("(d) 多尺度耦合", fontsize=9)
xx = np.linspace(0, 1, 500)
ax.plot(xx, 0.62 + 0.10*np.sin(2*np.pi*1.2*xx), color=GRAY, lw=1.6)
ax.plot(xx, 0.62 + 0.06*np.sin(2*np.pi*6*xx), color=BLUE, lw=1.2)
ax.plot(xx, 0.62 + 0.025*np.sin(2*np.pi*24*xx), color=RED, lw=1.0)
ax.text(0.06, 0.86, "长波背景强", fontsize=8, color=GRAY)
ax.text(0.06, 0.44, "5–20 km 短波弱", fontsize=8, color=RED)
ax.text(0.06, 0.30, "补短波不能破坏长波背景", fontsize=7.5, color=NAVY)
ax.annotate("", xy=(0.88, 0.50), xytext=(0.88, 0.82),
            arrowprops=dict(arrowstyle="<->", color=GOLD, lw=1.1))
ax.text(0.89, 0.63, "尺度\n分离", fontsize=7, color=GOLD)

fig.savefig(OUT / "fig_new_hard_4panel.png", bbox_inches="tight")
print("saved")
