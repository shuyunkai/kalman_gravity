"""KF 迭代过程图：输入、先验、1/3/6 轮、后验标准差。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from three_method_compare import load_region_data
from aniso_kf_test import build

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 10})

LON, LAT = (132.0, 152.0), (26.0, 42.0)


def main():
    d = load_region_data(LON, LAT, val_cruise="MR22-04_gravity_dmo", upscale=4)
    r = build(d["gravity_lr"].astype(float), d["bathy_15s"],
              d["lon_hr"], d["lat_hr"], w_keep=1.0, p0=2.0, n_iter=6,
              ship=d["ship_grav"], train=d["train_mask"],
              return_reconstructor=True)
    lon, lat = d["lon_hr"], d["lat_hr"]
    val = np.asarray(d["val_points"], float)
    ext = [lon[0], lon[-1], lat[0], lat[-1]]
    k = 4
    sub = (slice(k, -k), slice(k, -k))

    fields = [("(a) 输入：1′ + 船测", d["ssv_hr"]),
              ("(b) 先验：SSV + 地形", r.prior_2d),
              ("(c) 第 1 轮", r.x_history[0]),
              ("(d) 第 3 轮", r.x_history[2]),
              ("(e) 第 6 轮", r.x_history[5])]
    v = np.nanpercentile(np.abs(r.x_history[5][sub]), 98)

    fig, axes = plt.subplots(2, 3, figsize=(13.4, 5.2), dpi=220)
    for ax, (name, fld) in zip(axes.ravel()[:5], fields):
        im = ax.imshow(fld[sub], cmap="RdBu_r", origin="lower",
                       vmin=-v, vmax=v, extent=ext, aspect="auto")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("经度 (°E)", fontsize=9)
        ax.set_ylabel("纬度 (°N)", fontsize=9)
        ax.tick_params(labelsize=8)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("mGal", fontsize=8); cb.ax.tick_params(labelsize=7)
    axes[0, 0].scatter(val[::6, 0], val[::6, 1], s=0.6, c="k", alpha=0.5)

    ax = axes[1, 2]
    inc = r.x_history[5] - r.prior_2d
    vi = np.nanpercentile(np.abs(inc[sub]), 98)
    im = ax.imshow(inc[sub], cmap="PuOr_r", origin="lower",
                   vmin=-vi, vmax=vi, extent=ext, aspect="auto")
    ax.set_title("(f) 第 6 轮 − 先验", fontsize=11)
    ax.set_xlabel("经度 (°E)", fontsize=9)
    ax.set_ylabel("纬度 (°N)", fontsize=9)
    ax.tick_params(labelsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("mGal", fontsize=8); cb.ax.tick_params(labelsize=7)

    fig.tight_layout()
    p = OUT / "fig_new_iterations.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    main()
