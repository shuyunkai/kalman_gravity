"""15″ 应用需求页：应用表 + 1′/15″ 同区域短波对比。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
sys.path.insert(0, str(Path(__file__).parent))
from scale_decomposition import bandpass
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, RED, GOLD, LIGHT = "#0B2545", "#C0392B", "#C9A227", "#EEF3F9"
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


d = np.load(OUT / "validation_G_fields.npz")
lon, lat = d["lon"], d["lat"]
ssv, ckf = d["ssv"], d["ckf"]
dx = (lon[1]-lon[0])*111320*np.cos(np.radians(lat.mean()))
dy = (lat[1]-lat[0])*111320
selx = (lon >= 138.0) & (lon <= 142.0)
sely = (lat >= 33.5) & (lat <= 36.0)
x = lon[selx]; y = lat[sely]
ssv1 = gaussian_filter(ssv, 8e3/dx)
g1 = bandpass(ssv1, dx, dy, 5e3, 20e3)[np.ix_(sely, selx)]
g2 = bandpass(ckf, dx, dy, 5e3, 20e3)[np.ix_(sely, selx)]
v = np.nanpercentile(np.abs(np.r_[g1.ravel(), g2.ravel()]), 98)

fig = plt.figure(figsize=(13.4, 5.2), dpi=220)

ax = fig.add_axes([0.02, 0.10, 0.58, 0.82])
table(ax,
      [("重力匹配导航", "重力图匹配、惯导修正", "5–20 km", "连续短波梯度"),
       ("海底地形/工程", "海山、断裂、陡坡", "5–20 km", "边界与陡坡细节"),
       ("资源/浅层", "局部构造、基底、浅层异常", "5–20 km", "浅部局部细节")],
      ["应用", "典型用途", "需要波段", "15″ 能提供"],
      [0.20, 0.36, 0.17, 0.27], fs=8.5)

for pos, z, title in [([0.63, 0.53, 0.33, 0.36], g1, "(a) 1′ 产品（有效 12–20 km）：5–20 km"),
                      ([0.63, 0.10, 0.33, 0.36], g2, "(b) 15″ 场：5–20 km")]:
    ax = fig.add_axes(pos)
    im = ax.imshow(z, extent=[x[0], x[-1], y[0], y[-1]], origin="lower",
                   cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=5)

fig.text(0.795, 0.04, "同区域 5–20 km：15″ 保留更多短波细节",
         ha="center", fontsize=7.5, color=RED)

fig.savefig(OUT / "fig_new_15sec_compare.png", bbox_inches="tight")
print("saved")
