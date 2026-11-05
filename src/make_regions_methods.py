"""不同区域：真实地形 + 三种 15″ 算法对比。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 8})
regions = [("D_Nankai", "南海海槽"), ("F_PhilSea", "菲律宾海"), ("G_Japan", "日本海沟")]
fig, axes = plt.subplots(3, 4, figsize=(13.4, 5.8), dpi=200,
                         gridspec_kw={"wspace": 0.04, "hspace": 0.15})
ls = LightSource(azdeg=315, altdeg=45)
for i, (key, cn) in enumerate(regions):
    d = np.load(ROOT / f"output_real/three_parts/ship_regions/{key}.npz")
    lon, lat, ssv, bol, kf, bathy = d["lon"], d["lat"], d["ssv"], d["bol"], d["kf"], d["bathy"]
    step = max(1, min(ssv.shape)//380)
    ssv = ssv[::step, ::step]; bol = bol[::step, ::step]; kf = kf[::step, ::step]
    bathy = bathy[::step, ::step]; lon = lon[::step]; lat = lat[::step]
    ext = [lon[0], lon[-1], lat[0], lat[-1]]
    v = np.nanpercentile(np.abs(kf), 98)
    ax = axes[i, 0]
    rgb = ls.shade(bathy, cmap=plt.cm.Greys, vert_exag=0.5, blend_mode="overlay")
    ax.imshow(rgb, origin="lower", extent=ext, aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    if i == 0: ax.set_title("地形", fontsize=9)
    if i == 0: ax.set_ylabel(cn, fontsize=8)
    for j, (z, title) in enumerate([(ssv, "双线性 15″"), (bol, "地形先验 KF"), (kf, "相干保持 KF")], start=1):
        ax = axes[i, j]
        ax.imshow(z, cmap="RdBu_r", origin="lower", extent=ext,
                  vmin=-v, vmax=v, aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0: ax.set_title(title, fontsize=9)
fig.savefig(OUT / "fig_new_regions_methods.png", bbox_inches="tight")
print("saved")
