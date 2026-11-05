"""第4节：三个应用场景下 1′ 与 15″ 的地图对比。"""
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
                     "axes.unicode_minus": False, "font.size": 8})
d = np.load(OUT / "validation_G_fields.npz")
lon, lat = d["lon"], d["lat"]
ssv = gaussian_filter(d["ssv"], 8e3/((lon[1]-lon[0])*111320*np.cos(np.radians(lat.mean()))))
ckf = d["ckf"]
dx = (lon[1]-lon[0])*111320*np.cos(np.radians(lat.mean()))
dy = (lat[1]-lat[0])*111320
sx = (lon >= 138) & (lon <= 142); sy = (lat >= 33.5) & (lat <= 36)
ext = [lon[sx][0], lon[sx][-1], lat[sy][0], lat[sy][-1]]
g1 = bandpass(ssv, dx, dy, 5e3, 20e3)[np.ix_(sy, sx)]
g2 = bandpass(ckf, dx, dy, 5e3, 20e3)[np.ix_(sy, sx)]
# navigation uses gradient magnitude
gy1, gx1 = np.gradient(g1); gy2, gx2 = np.gradient(g2)
n1 = np.hypot(gx1, gy1); n2 = np.hypot(gx2, gy2)
cols = [("导航：短波梯度", n1, n2), ("地形/工程：5–20 km", g1, g2),
        ("资源/浅层：5–20 km", g1, g2)]
fig, axes = plt.subplots(2, 3, figsize=(13.4, 5.2), dpi=200,
                         gridspec_kw={"wspace": 0.05, "hspace": 0.12})
for j, (title, a, b) in enumerate(cols):
    for i, z in enumerate([a, b]):
        ax = axes[i, j]
        v = np.nanpercentile(np.abs(z), 98)
        im = ax.imshow(z, cmap="RdBu_r", extent=ext, origin="lower",
                       vmin=-v, vmax=v, aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_title(title, fontsize=8)
    axes[0, 0].set_ylabel("1′ 产品", fontsize=9)
    axes[1, 0].set_ylabel("15″ 场", fontsize=9)
fig.savefig(OUT / "fig_new_apps_1_vs_15.png", bbox_inches="tight")
print("saved")
