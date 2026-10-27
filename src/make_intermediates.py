"""中间变量页：C、wC^2、预测修正、船测更新增量。"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from three_method_compare import load_region_data
import aniso_kf_test as ak
from aniso_kf_test import build
LON, LAT = (132.0, 152.0), (26.0, 42.0)
ak.LON, ak.LAT = LON, LAT

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})

data = load_region_data(LON, LAT, val_cruise="MR22-04_gravity_dmo", upscale=4)
lon, lat = data["lon_hr"], data["lat_hr"]
r = build(data["gravity_lr"].astype(float), data["bathy_15s"], lon, lat,
          w_keep=1.0, p0=2.0, n_iter=6, ship=data["ship_grav"],
          train=data["train_mask"], return_reconstructor=True)

C = r.coh_2d
W = r.weight_2d
pred = np.sum(r.pred_corr_history, axis=0)
ship = np.sum(r.ship_inc_history, axis=0)
direct = data["train_mask"] & np.isfinite(data["ship_grav"])
ship_plot = ship.copy()
ship_plot[direct] = 0.0

k = 4
sub = (slice(k, -k), slice(k, -k))
ext = [lon[0], lon[-1], lat[0], lat[-1]]

fig, axes = plt.subplots(2, 2, figsize=(12.8, 5.4), dpi=220)
panels = [
    ("(a) 相干 $C$", C, "viridis", None, "$C$"),
    ("(b) 权重 $wC^2$", W, "magma", None, "$wC^2$"),
    ("(c) 预测修正 $\\sum wC^2(x-G_\\sigma x)$", pred, "RdBu_r", "sym", "mGal"),
    ("(d) 船测更新邻域扩散（去测线点）", ship_plot, "RdBu_r", "sym", "mGal"),
]
for ax, (title, fld, cmap, mode, unit) in zip(axes.ravel(), panels):
    z = fld[sub]
    if mode == "sym":
        v = np.nanpercentile(np.abs(z), 98)
        im = ax.imshow(z, cmap=cmap, origin="lower", vmin=-v, vmax=v,
                       extent=ext, aspect="auto")
    else:
        vmax = 1.0 if title.startswith("(a)") else np.nanpercentile(z, 98)
        im = ax.imshow(z, cmap=cmap, origin="lower", vmin=0, vmax=vmax,
                       extent=ext, aspect="auto")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("经度 (°E)", fontsize=7); ax.set_ylabel("纬度 (°N)", fontsize=7)
    ax.tick_params(labelsize=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(unit, fontsize=7); cb.ax.tick_params(labelsize=6)

fig.tight_layout()
fig.savefig(OUT / "fig_new_intermediates.png", bbox_inches="tight")
print("saved")
