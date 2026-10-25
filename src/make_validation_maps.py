"""验证页 2×2 大图：三张场 + 差值。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 10})


def main():
    d = np.load(OUT / "validation_G_fields.npz")
    lon, lat = d["lon"], d["lat"]
    ssv, kf, ckf = d["ssv"], d["kf"], d["ckf"]
    ext = [lon[0], lon[-1], lat[0], lat[-1]]
    k = 4
    sub = (slice(k, -k), slice(k, -k))
    v = np.nanpercentile(np.abs(ckf[sub]), 98)

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 5.0), dpi=220)
    for ax, name, fld in ((axes[0, 0], "(a) 1′ 产品", ssv),
                          (axes[0, 1], "(b) 标准 KF", kf),
                          (axes[1, 0], "(c) 相干保持 KF", ckf)):
        im = ax.imshow(fld[sub], cmap="RdBu_r", origin="lower",
                       vmin=-v, vmax=v, extent=ext, aspect="auto")
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("经度 (°E)", fontsize=10)
        ax.set_ylabel("纬度 (°N)", fontsize=10)
        ax.tick_params(labelsize=9)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("mGal", fontsize=9); cb.ax.tick_params(labelsize=8)

    ax = axes[1, 1]
    diff = ckf - ssv
    vd = np.nanpercentile(np.abs(diff[sub]), 98)
    im = ax.imshow(diff[sub], cmap="PuOr_r", origin="lower", vmin=-vd, vmax=vd,
                   extent=ext, aspect="auto")
    ax.set_title("(d) 相干保持 KF − 1′ 差值", fontsize=12)
    ax.set_xlabel("经度 (°E)", fontsize=10)
    ax.set_ylabel("纬度 (°N)", fontsize=10)
    ax.tick_params(labelsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("mGal", fontsize=9); cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    p = OUT / "fig_new_validation_maps.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    main()
