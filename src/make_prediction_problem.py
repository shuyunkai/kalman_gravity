"""标准 KF 预测步问题：各向同性平滑 vs 结构感知。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, RED, GOLD = "#0B2545", "#C0392B", "#C9A227"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 7})

fig = plt.figure(figsize=(13.4, 4.2), dpi=240)

# (a) isotropic kernel across ridge
ax = fig.add_axes([0.03, 0.16, 0.28, 0.72])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
for i in range(9):
    ax.plot([0.08, 0.92], [0.08+i*0.105]*2, color="#D5DCE5", lw=0.6)
for j in range(9):
    ax.plot([0.08+j*0.105]*2, [0.08, 0.92], color="#D5DCE5", lw=0.6)
ax.plot([0.1, 0.9], [0.1, 0.9], color=GOLD, lw=7, alpha=0.55)
ax.add_patch(Circle((0.5, 0.5), 0.25, fill=False, ec=RED, lw=1.5))
ax.annotate("", xy=(0.5, 0.83), xytext=(0.5, 0.5),
            arrowprops=dict(arrowstyle="<->", color=RED, lw=1.2))
ax.annotate("", xy=(0.83, 0.5), xytext=(0.5, 0.5),
            arrowprops=dict(arrowstyle="<->", color=RED, lw=1.2))
ax.text(0.5, 1.02, "(a) 各向同性预测", ha="center", fontsize=8, color=NAVY, fontweight="bold")
ax.text(0.5, -0.04, "脊线两侧被同样平滑", ha="center", fontsize=7, color=RED)

# (b) structure-aware kernel
ax = fig.add_axes([0.35, 0.16, 0.28, 0.72])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
for i in range(9):
    ax.plot([0.08, 0.92], [0.08+i*0.105]*2, color="#D5DCE5", lw=0.6)
for j in range(9):
    ax.plot([0.08+j*0.105]*2, [0.08, 0.92], color="#D5DCE5", lw=0.6)
ax.plot([0.1, 0.9], [0.1, 0.9], color=GOLD, lw=7, alpha=0.55)
ax.add_patch(Ellipse((0.5, 0.5), 0.22, 0.50, angle=45, fill=False, ec=RED, lw=1.5))
ax.text(0.5, 1.02, "(b) 结构感知预测", ha="center", fontsize=8, color=NAVY, fontweight="bold")
ax.text(0.5, -0.04, "沿脊线平滑，不跨脊线", ha="center", fontsize=7, color=RED)

# (c) profile
ax = fig.add_axes([0.68, 0.19, 0.29, 0.66])
x = np.linspace(-10, 10, 601)
truth = 45*np.exp(-(x/2.0)**2) + 15*np.exp(-((x-4)/3.0)**2)
std = gaussian_filter1d(truth, 4.2)
ckf = 0.93*truth
ax.plot(x, truth, color="k", lw=1.5, label="真实/船测")
ax.plot(x, std, color="#7F8C8D", lw=1.3, ls="--", label="标准 KF 预测")
ax.plot(x, ckf, color=RED, lw=1.3, label="结构感知预测")
ax.set_xlabel("跨脊线距离 (km)", fontsize=7)
ax.set_ylabel("重力异常 (mGal)", fontsize=7)
ax.set_title("(c) 跨脊线剖面", fontsize=8, color=NAVY, fontweight="bold")
ax.legend(fontsize=6, frameon=False, loc="upper right")
ax.tick_params(labelsize=6)
ax.grid(alpha=0.2)
fig.savefig(OUT / "fig_prediction_problem.png", bbox_inches="tight")
print("saved")
