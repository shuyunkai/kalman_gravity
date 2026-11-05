"""新增页：15″ 重力场的应用需求与技术条件。"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, RED, GOLD, BLUE, LIGHT = "#0B2545", "#C0392B", "#C9A227", "#7FA8C9", "#EEF3F9"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})


def table(ax, rows, cols, widths, fs=8.5):
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    h = 1.0/(len(rows)+1); y = 1.0; x = 0
    for c, w in zip(cols, widths):
        ax.add_patch(Rectangle((x, y-h), w, h, fc=NAVY, ec="white", lw=1))
        ax.text(x+w/2, y-h/2, c, ha="center", va="center", color="white",
                fontsize=fs, fontweight="bold")
        x += w
    y -= h
    for r in rows:
        x = 0
        for val, w in zip(r, widths):
            ax.add_patch(Rectangle((x, y-h), w, h, fc=LIGHT, ec="white", lw=1))
            ax.text(x+0.012, y-h/2, val, ha="left", va="center",
                    fontsize=fs-0.5, color="#1A1A1A")
            x += w
        y -= h


fig = plt.figure(figsize=(13.4, 5.2), dpi=220)

ax = fig.add_axes([0.02, 0.10, 0.60, 0.82])
table(ax,
      [("重力匹配导航", "重力图匹配、惯导修正", "5–20 km", "连续短波梯度"),
       ("海底地形/工程", "海山、断裂、陡坡", "5–20 km", "边界与陡坡细节"),
       ("资源/浅层", "局部构造、基底、浅层异常", "5–20 km", "浅部局部细节")],
      ["应用", "典型用途", "需要波段", "15″ 能提供"],
      [0.20, 0.36, 0.17, 0.27], fs=8.5)

ax = fig.add_axes([0.65, 0.10, 0.33, 0.82]); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.text(0.5, 0.96, "技术条件", ha="center", fontsize=10, color=NAVY,
        fontweight="bold")
items = [("15″ 格距", "≈ 0.46 km"), ("最短波长", "λmin ≈ 0.93 km"),
         ("目标频带", "5–20 km")]
for i, (k, v) in enumerate(items):
    y = 0.74 - i*0.18
    ax.add_patch(FancyBboxPatch((0.06, y), 0.88, 0.13,
                                boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc="#FFF8E7", ec=GOLD, lw=0.9))
    ax.text(0.10, y+0.065, k, fontsize=8, color=NAVY, va="center")
    ax.text(0.92, y+0.065, v, fontsize=9, color=RED, va="center", ha="right",
            fontweight="bold")

# band coverage
ax = fig.add_axes([0.66, 0.08, 0.31, 0.13])
ax.set_xscale("log"); ax.set_xlim(1, 100); ax.set_ylim(0, 1)
ax.axvspan(5, 20, color=RED, alpha=0.15)
ax.axvline(0.93, color=BLUE, lw=1.2, ls="--")
ax.text(1.1, 0.62, "15″\n0.93", fontsize=6, color=BLUE)
ax.text(12, 0.20, "5–20 km", ha="center", fontsize=7, color=RED)
ax.set_yticks([]); ax.set_xlabel("波长 (km)", fontsize=6.5)
ax.tick_params(labelsize=5.5)

fig.savefig(OUT / "fig_new_15sec_needs.png", bbox_inches="tight")
print("saved")
