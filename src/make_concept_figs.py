"""概念图：断裂识别的重力响应 + 海底地形反演原理。"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle, FancyArrowPatch

FONT = Path(r"D:\gravity\fonts\SourceHanSansSC-Regular.otf")
if FONT.exists():
    fm.fontManager.addfont(str(FONT))
plt.rcParams["font.sans-serif"] = ["Source Han Sans SC", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(r"D:\gravity\kalman_gravity\output_real\three_parts")

# ---------------- 图 1：断层与重力响应
fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), dpi=200)

# (a) 剖面：断层错断密度层
ax = axes[0]
colors = ["#D6E4F0", "#9DC3E6", "#2E75B6", "#1F3864"]
for i, c in enumerate(colors):
    ax.add_patch(Rectangle((0, i), 0.48, 1, color=c, ec="white", lw=0.8))
    ax.add_patch(Rectangle((0.52, i + 0.35), 0.48, 1, color=c, ec="white", lw=0.8))
ax.plot([0.48, 0.52], [0, 4.35], color="#C0392B", lw=2.2)
ax.annotate("断层", xy=(0.5, 3.6), xytext=(0.62, 3.9), color="#C0392B",
            fontsize=9, arrowprops=dict(arrowstyle="->", color="#C0392B"))
ax.set_xlim(0, 1); ax.set_ylim(0, 4.4)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("(a) 断层错断密度界面", fontsize=10)

# (b) 重力剖面：梯度带
ax = axes[1]
x = np.linspace(-3, 3, 300)
g = 40 / (1 + np.exp(-2.2 * x)) - 20
ax.plot(x, g, color="#1F3864", lw=1.8)
ax.axvline(0, color="#C0392B", ls="--", lw=1.2)
ax.set_xlabel("距离", fontsize=8); ax.set_ylabel("重力异常", fontsize=8)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("(b) 重力梯度带", fontsize=10)
ax.grid(alpha=0.25)

# (c) 平面：线性体
ax = axes[2]
Y, X = np.mgrid[-1:1:200j, -1.5:1.5:300j]
f = np.exp(-((X - 0.3 * Y) ** 2) / 0.02)
ax.imshow(f, cmap="RdBu_r", extent=[-1.5, 1.5, -1, 1], origin="lower")
ax.plot([-1.5, 1.5], [-0.5, 0.5], color="#1F3864", lw=1.2, ls="--")
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("(c) 线性体提取", fontsize=10)
fig.suptitle("断裂在重力场上的表现：密度界面错断 → 梯度带 → 线性体",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig_fault_concept.png", bbox_inches="tight")
plt.close(fig)

# ---------------- 图 2：海底地形反演
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.2), dpi=200)

# (a) 剖面：海水/地壳
ax = axes[0]
x = np.linspace(0, 10, 400)
topo = 3.0 + 0.5 * np.sin(1.2 * x) + 0.2 * np.sin(3.1 * x)
ax.fill_between(x, 4.2, topo, color="#BDD7EE", label="海水 $\\rho_w$=1030")
ax.fill_between(x, topo, 0, color="#8C6D4F", label="地壳 $\\rho_c$=2670")
ax.plot(x, topo, color="#1F3864", lw=1.8)
ax.set_ylim(0, 4.2); ax.set_xlim(0, 10)
ax.set_xticks([]); ax.set_yticks([])
ax.legend(fontsize=7.5, frameon=False, loc="lower left")
ax.set_title("(a) 海底地形与密度差 $\\Delta\\rho$=1640 kg/m$^3$", fontsize=10)

# (b) 重力异常（界面起伏的一阶响应）
ax = axes[1]
g = 0.8 * np.sin(1.2 * x) + 0.25 * np.sin(3.1 * x)
ax.plot(x, g, color="#C0392B", lw=1.8)
ax.set_xlabel("距离", fontsize=8); ax.set_ylabel("重力异常 (mGal)", fontsize=8)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("(b) 对应的重力异常", fontsize=10)
ax.grid(alpha=0.25)
fig.suptitle("海底地形反演：由重力异常恢复密度界面起伏（Parker–Oldenburg）",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig_inversion_concept.png", bbox_inches="tight")
plt.close(fig)
print("saved")
