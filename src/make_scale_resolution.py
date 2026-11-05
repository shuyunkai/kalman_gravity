"""过渡页：重力异常—波长—分辨率。"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT = r"output_real/three_parts/newdeck"
NAVY, GOLD, RED, BLUE, GRAY = "#0B2545", "#C9A227", "#C0392B", "#7FA8C9", "#90A4AE"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})

fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.6), dpi=220)

# (a) Δg = g_obs - gamma
ax = axes[0]
x = np.linspace(0, 1, 500)
gamma = 0.25 * np.sin(2*np.pi*(x-0.1)) + 0.35
g = gamma + 0.10*np.sin(2*np.pi*9*x) + 0.06*np.exp(-((x-0.58)/0.10)**2)
ax.plot(x, gamma, color=GRAY, lw=1.6, ls="--", label="正常场 γ（理论）")
ax.plot(x, g, color=NAVY, lw=1.6, label="实测 g")
ax.fill_between(x, gamma, g, color=RED, alpha=0.18)
ax.annotate("Δg = g − γ", xy=(0.58, g[290]), xytext=(0.56, 0.78),
            color=RED, fontsize=12, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1))
ax.set_xlim(0, 1); ax.set_ylim(0.0, 0.9)
ax.set_title("(a) 重力异常是实测减正常场", fontsize=10)
ax.set_xlabel("位置", fontsize=8); ax.set_ylabel("重力", fontsize=8)
ax.legend(fontsize=7, frameon=False, loc="lower right")
ax.set_xticks([]); ax.set_yticks([])
ax.grid(alpha=0.15)

# (b) wavelength decomposition
ax = axes[1]
x = np.linspace(0, 1, 800)
long = 0.30*np.sin(2*np.pi*1.2*x)
mid = 0.16*np.sin(2*np.pi*7*x + 0.6)
short = 0.07*np.sin(2*np.pi*22*x + 1.1)
ax.plot(x, long, color=GRAY, lw=1.3, label="长波")
ax.plot(x, mid, color=BLUE, lw=1.3, label="中波")
ax.plot(x, short, color=RED, lw=1.3, label="短波")
ax.plot(x, long+mid+short, color=NAVY, lw=1.8, label="叠加场")
ax.annotate("", xy=(0.307, 0.55), xytext=(0.165, 0.55),
            arrowprops=dict(arrowstyle="<->", color=RED, lw=1.0))
ax.text(0.236, 0.58, "波长 λ = 相邻波峰距离", ha="center", fontsize=7.5,
        color=RED)
ax.set_xlim(0, 1); ax.set_ylim(-0.65, 0.65)
ax.set_title("(b) 重力场按波长分解", fontsize=10)
ax.set_xlabel("位置", fontsize=8); ax.set_yticks([])
ax.legend(fontsize=7, frameon=False, ncol=2, loc="lower center")
ax.grid(alpha=0.15)

# (c) grid / resolution
ax = axes[2]
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.add_patch(Rectangle((0.05, 0.38), 0.42, 0.42, fill=False, ec=NAVY, lw=1.8))
for i in range(1, 4):
    ax.plot([0.05+0.105*i, 0.05+0.105*i], [0.38, 0.80], color=NAVY, lw=0.8)
    ax.plot([0.05, 0.47], [0.38+0.105*i, 0.38+0.105*i], color=NAVY, lw=0.8)
ax.text(0.26, 0.84, "1′ 格网 = 4×4 个 15″ 格网", ha="center", fontsize=8.5,
        color=NAVY, fontweight="bold")
ax.text(0.56, 0.72, "1′  Δx ≈ 1.85 km", fontsize=8, color=NAVY)
ax.text(0.56, 0.64, "15″  Δx ≈ 0.46 km", fontsize=8, color=NAVY)
ax.text(0.56, 0.54, "分辨率：λ_min = 2Δx", fontsize=8, color=GOLD,
        fontweight="bold")
ax.text(0.56, 0.46, "1′  λ_min ≈ 3.7 km", fontsize=8, color="#44546A")
ax.text(0.56, 0.38, "15″  λ_min ≈ 0.93 km", fontsize=8, color="#44546A")
ax.text(0.56, 0.26, "1′ 产品有效约 12–20 km", fontsize=8, color=RED,
        fontweight="bold")
ax.text(0.56, 0.18, "目标 5–20 km 落在 1′ 之外", fontsize=8, color=RED)
ax.text(0.56, 0.10, "15″ 格网只为表示短波提供可能，\n能否恢复取决于观测与约束", fontsize=7.2,
        color="#44546A")

fig.tight_layout()
fig.savefig(f"{ROOT}/fig_new_scale_resolution.png", bbox_inches="tight")
print("saved")
