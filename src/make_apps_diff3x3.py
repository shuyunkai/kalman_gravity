"""应用对比 3×3：1′、15″、差值 × 导航/地形/资源。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
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
dx = (lon[1]-lon[0])*111320*np.cos(np.radians(lat.mean()))
dy = (lat[1]-lat[0])*111320
ssv = gaussian_filter(d["ssv"], 8e3/dx)
ckf = d["ckf"]
sx = (lon >= 138) & (lon <= 142); sy = (lat >= 33.5) & (lat <= 36)
ext = [lon[sx][0], lon[sx][-1], lat[sy][0], lat[sy][-1]]
g1 = bandpass(ssv, dx, dy, 5e3, 20e3)[np.ix_(sy, sx)]
g2 = bandpass(ckf, dx, dy, 5e3, 20e3)[np.ix_(sy, sx)]
# navigation gradient
gy1, gx1 = np.gradient(g1); gy2, gx2 = np.gradient(g2)
n1 = np.hypot(gx1, gy1); n2 = np.hypot(gx2, gy2)
# resource: vertical derivative of 5-20 band
r1 = np.gradient(g1, axis=0); r2 = np.gradient(g2, axis=0)
cols = [("导航：5–20 km 梯度", n1, n2, "Reds"),
        ("地形/工程：5–20 km", g1, g2, "RdBu_r"),
        ("资源/浅层：5–20 km 垂向梯度", r1, r2, "PRGn")]
fig, axes = plt.subplots(2, 3, figsize=(13.4, 4.6), dpi=200,
                         gridspec_kw={"wspace": 0.04, "hspace": 0.12})
for j, (title, a, b, cmap) in enumerate(cols):
    for i, z in enumerate([a, b]):
        ax = axes[i, j]
        v = np.nanpercentile(np.abs(z), 98)
        if "导航" in title:
            im = ax.imshow(z, cmap="Reds", extent=ext, origin="lower",
                           vmin=0, vmax=v, aspect="auto")
        else:
            im = ax.imshow(z, cmap=cmap, extent=ext, origin="lower",
                           vmin=-v, vmax=v, aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_title(title, fontsize=8)
    axes[0, 0].set_ylabel("1′ 产品", fontsize=8)
    axes[1, 0].set_ylabel("15″ 场", fontsize=8)
fig.subplots_adjust(right=0.82)
for k, (cmap, label) in enumerate([("Reds", "浅→深\n梯度小→大"),
                                    ("RdBu_r", "蓝→白→红\n负→0→正"),
                                    ("PRGn", "紫→白→绿\n负→0→正")]):
    cax = fig.add_axes([0.86, 0.70-0.22*k, 0.015, 0.16])
    ColorbarBase(cax, cmap=plt.get_cmap(cmap), norm=Normalize(0, 1))
    cax.set_xticks([]); cax.set_yticks([])
    cax.set_ylabel(label, fontsize=6, rotation=0, labelpad=18, va="center")
fig.text(0.5, -0.015, "红/蓝=重力异常正/负；导航、资源为梯度（mGal/km）；区域 138–142°E, 33.5–36°N，与上一页地形图同范围", ha="center", fontsize=6.5, color="#333333")
fig.savefig(OUT / "fig_new_apps_diff3x3.png", bbox_inches="tight")
print("saved")
