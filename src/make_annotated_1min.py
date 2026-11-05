"""1′ 产品局限：局部标注图（不出现 15″ 对比）。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).parent))
from scale_decomposition import bandpass
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})
d = np.load(OUT / "validation_G_fields.npz")
lon, lat = d["lon"], d["lat"]
dx = (lon[1]-lon[0])*111320*np.cos(np.radians(lat.mean()))
dy = (lat[1]-lat[0])*111320
ssv = gaussian_filter(d["ssv"], 8e3/dx)
sx = (lon >= 138) & (lon <= 142); sy = (lat >= 33.5) & (lat <= 36)
z = ssv[np.ix_(sy, sx)]
ext = [lon[sx][0], lon[sx][-1], lat[sy][0], lat[sy][-1]]

fig, ax = plt.subplots(figsize=(13.4, 4.4), dpi=220)
im = ax.imshow(z, extent=ext, origin="lower", cmap="RdBu_r",
               vmin=-np.nanpercentile(np.abs(z), 98),
               vmax=np.nanpercentile(np.abs(z), 98), aspect="auto")
ax.set_xlabel("经度 (°E)", fontsize=8); ax.set_ylabel("纬度 (°N)", fontsize=8)
ax.set_title("1′ 产品：局部细节被平滑", fontsize=10)
ax.tick_params(labelsize=7)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("mGal", fontsize=7); cb.ax.tick_params(labelsize=6)

ann = [(139.0, 35.1, "海山边界模糊"),
       (140.4, 34.4, "断裂细节缺失"),
       (141.1, 35.5, "短波梯度弱")]
for x, y, txt in ann:
    ax.plot(x, y, "o", color="#C0392B", ms=5)
    ax.annotate(txt, xy=(x, y), xytext=(x-0.6, y+0.45),
                fontsize=8, color="#C0392B",
                arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.0))
fig.savefig(OUT / "fig_new_1min_limit.png", bbox_inches="tight")
print("saved")
