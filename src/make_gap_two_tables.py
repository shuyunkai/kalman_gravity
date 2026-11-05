"""第3页：1′ 无法满足的场景 + 1′ 产品自身缺陷。"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent))
from scale_decomposition import bandpass
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Rectangle

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, RED, LIGHT, GRAY = "#0B2545", "#C0392B", "#EEF3F9", "#90A4AE"
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

ax = fig.add_axes([0.02, 0.34, 0.56, 0.58])
table(ax,
      [("重力匹配导航", "5–20 km", "20 km 以下被平滑，短波梯度缺失"),
       ("海底地形/工程", "5–20 km", "海山/断裂边界模糊，陡坡细节丢失"),
       ("资源/浅层", "5–20 km", "浅部细节缺失")],
      ["无法满足的场景", "需要波长", "具体表现"],
      [0.24, 0.17, 0.59], fs=8)
ax.set_title("(a) 哪些场景 1′ 产品无法满足", fontsize=9, pad=4)

ax = fig.add_axes([0.62, 0.34, 0.36, 0.58])
table(ax,
      [("SSV33.1", "20 km 以下被平滑"),
       ("DTU21GRA", "近岸/极区精度下降"),
       ("SDUST2022GRA", "激光脚点稀疏，仅沿轨条带"),
       ("NSOAS24", "坡度差分放大沿轨噪声")],
      ["1′ 产品", "主要缺点"],
      [0.34, 0.66], fs=8)
ax.set_title("(b) 各 1′ 产品的具体缺点", fontsize=9, pad=4)

d = np.load(OUT / "validation_G_fields.npz")
lon, lat = d["lon"], d["lat"]
dx = (lon[1]-lon[0])*111320*np.cos(np.radians(lat.mean()))
dy = (lat[1]-lat[0])*111320
ssv = gaussian_filter(d["ssv"], 8e3/dx)
g1 = bandpass(ssv, dx, dy, 5e3, 20e3)
g2 = bandpass(d["ckf"], dx, dy, 5e3, 20e3)
sx = (lon >= 138) & (lon <= 142)
sy = (lat >= 33.5) & (lat <= 36)
z1 = g1[np.ix_(sy, sx)]; z2 = g2[np.ix_(sy, sx)]
ext = [lon[sx][0], lon[sx][-1], lat[sy][0], lat[sy][-1]]
v = np.nanpercentile(np.abs(np.r_[z1.ravel(), z2.ravel()]), 98)
for pos, z, title in [([0.07, 0.02, 0.38, 0.25], z1, "(a) 1′ 产品：边界/细节模糊"),
                      ([0.54, 0.02, 0.38, 0.25], z2, "(b) 15″ 场：边界/细节清晰")]:
    ax = fig.add_axes(pos)
    im = ax.imshow(z, extent=ext, origin="lower", cmap="RdBu_r",
                   vmin=-v, vmax=v, aspect="auto")
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=5)

fig.savefig(OUT / "fig_new_gap_two_tables.png", bbox_inches="tight")
print("saved")
