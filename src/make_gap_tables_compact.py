"""第4页紧凑表：两张表横排，一条。"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, LIGHT = "#0B2545", "#EEF3F9"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False})


def table(ax, rows, cols, widths, fs=6.6):
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    h = 1.0/(len(rows)+1); y = 1.0; x = 0
    for c, w in zip(cols, widths):
        ax.add_patch(Rectangle((x, y-h), w, h, fc=NAVY, ec="white", lw=0.8))
        ax.text(x+w/2, y-h/2, c, ha="center", va="center", color="white",
                fontsize=fs, fontweight="bold")
        x += w
    y -= h
    for r in rows:
        x = 0
        for val, w in zip(r, widths):
            ax.add_patch(Rectangle((x, y-h), w, h, fc=LIGHT, ec="white", lw=0.8))
            ax.text(x+0.008, y-h/2, val, ha="left", va="center",
                    fontsize=fs-0.3, color="#1A1A1A")
            x += w
        y -= h


fig = plt.figure(figsize=(13.4, 1.55), dpi=260)
ax = fig.add_axes([0.01, 0.10, 0.56, 0.80])
table(ax,
      [("重力匹配导航", "5–20 km", "20 km 以下被平滑，短波梯度缺失"),
       ("海底地形/工程", "5–20 km", "海山/断裂边界模糊，陡坡细节丢失"),
       ("资源/浅层", "5–20 km", "浅部细节缺失")],
      ["无法满足的场景", "需要波长", "具体表现"],
      [0.24, 0.16, 0.60], fs=6.4)
ax.set_title("(a) 哪些场景 1′ 产品无法满足", fontsize=7, pad=2)

ax = fig.add_axes([0.60, 0.10, 0.39, 0.80])
table(ax,
      [("SSV33.1", "20 km 以下被平滑"),
       ("DTU21GRA", "近岸/极区精度下降"),
       ("SDUST2022GRA", "激光脚点稀疏"),
       ("NSOAS24", "坡度差分放大沿轨噪声")],
      ["1′ 产品", "主要缺点"],
      [0.38, 0.62], fs=6.4)
ax.set_title("(b) 各 1′ 产品的具体缺点", fontsize=7, pad=2)
fig.savefig(OUT / "fig_gap_tables_compact.png", bbox_inches="tight")
print("saved")
