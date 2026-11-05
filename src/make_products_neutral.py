"""1′ 产品介绍页：只讲产品、覆盖、有效分辨率。"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, GOLD, LIGHT = "#0B2545", "#C9A227", "#EEF3F9"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})

fig, ax = plt.subplots(figsize=(13.4, 5.0), dpi=220)
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

cols = ["产品", "来源/方法", "覆盖", "有效分辨率"]
widths = [0.15, 0.32, 0.14, 0.39]
rows = [
    ("SSV33.1", "测高重力", "全球", "1′ 格网，有效约 12–20 km"),
    ("DTU21GRA", "测高重力", "全球", "1′ 格网"),
    ("SDUST2022GRA", "雷达 + 激光测高", "全球", "1′ 格网"),
    ("NSOAS24", "全球海洋重力产品", "全球", "1′ 格网"),
]
y0, h = 0.78, 0.115
x = 0.02
for c, w in zip(cols, widths):
    ax.add_patch(Rectangle((x, y0), w, h, fc=NAVY, ec="white", lw=1))
    ax.text(x+w/2, y0+h/2, c, ha="center", va="center", color="white",
            fontsize=10, fontweight="bold")
    x += w
for i, r in enumerate(rows):
    y = y0 - (i+1)*h
    x = 0.02
    for val, w in zip(r, widths):
        ax.add_patch(Rectangle((x, y), w, h, fc=LIGHT if i % 2 == 0 else "white",
                               ec="white", lw=1))
        ax.text(x+0.012, y+h/2, val, ha="left", va="center", fontsize=9)
        x += w

# bottom neutral facts
facts = ["全球覆盖", "1′ 格网发布", "有效分辨率约 12–20 km"]
for i, f in enumerate(facts):
    x = 0.04 + i*0.32
    ax.add_patch(FancyBboxPatch((x, 0.12), 0.28, 0.13,
                                boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc="#FFF8E7", ec=GOLD, lw=1))
    ax.text(x+0.14, 0.185, f, ha="center", va="center", fontsize=10,
            color=NAVY, fontweight="bold")
ax.text(0.5, 0.03, "这些产品提供长波/中波背景；短波能力需结合具体产品与应用判断",
        ha="center", fontsize=8, color="#44546A")

fig.tight_layout()
fig.savefig(OUT / "fig_new_products_neutral.png", bbox_inches="tight")
print("saved")
