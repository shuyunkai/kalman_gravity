"""验证页：局部放大 1′/CKF/差值 + 密度散点。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, PALE = "#0B2545", "#D6E2F0"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})


def main():
    d = np.load(OUT / "validation_G_fields.npz")
    lon, lat = d["lon"], d["lat"]
    ssv, ckf = d["ssv"], d["ckf"]
    val = np.asarray(d["val"], float)
    ix = np.where((lon >= 142) & (lon <= 145))[0]
    iy = np.where((lat >= 34) & (lat <= 37))[0]
    j0, j1 = ix[0], ix[-1]
    i0, i1 = iy[0], iy[-1]
    ext = [lon[j0], lon[j1], lat[i0], lat[i1]]

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 5.4), dpi=220)
    for ax, name, fld in ((axes[0, 0], "(a) 双线性 15″ 场（SSV，日本区）", ssv),
                          (axes[0, 1], "(b) 相干保持 KF（日本区）", ckf)):
        z = fld[i0:i1, j0:j1]
        v = np.nanpercentile(np.abs(z), 98)
        im = ax.imshow(z, cmap="RdBu_r", origin="lower", vmin=-v, vmax=v,
                       extent=ext, aspect="auto")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("经度 (°E)", fontsize=8); ax.set_ylabel("纬度 (°N)", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("mGal", fontsize=7)
    ax = axes[1, 0]
    diff = (ckf - ssv)[i0:i1, j0:j1]
    vd = np.nanpercentile(np.abs(diff), 98)
    im = ax.imshow(diff, cmap="PuOr_r", origin="lower", vmin=-vd, vmax=vd,
                   extent=ext, aspect="auto")
    ax.set_title("(c) 相干保持 KF − 双线性 15″ 场（日本区）", fontsize=10)
    ax.set_xlabel("经度 (°E)", fontsize=8); ax.set_ylabel("纬度 (°N)", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("mGal", fontsize=7)

    ax = axes[1, 1]
    ixv = np.clip(np.searchsorted(lon, val[:, 0]), 0, len(lon)-1)
    iyv = np.clip(np.searchsorted(lat, val[:, 1]), 0, len(lat)-1)
    hb = ax.hexbin(val[:, 2], ckf[iyv, ixv], gridsize=60, cmap="magma",
                   mincnt=1, bins="log")
    lo, hi = np.percentile(val[:, 2], [1, 99])
    ax.plot([lo, hi], [lo, hi], "--", color="cyan", lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("船测重力 (mGal)", fontsize=8)
    ax.set_ylabel("相干保持 KF (mGal)", fontsize=8)
    ax.set_title("(d) 船测—模型密度散点", fontsize=10)
    ax.tick_params(labelsize=7)
    fig.colorbar(hb, ax=ax, fraction=0.04, pad=0.02).set_label("log10(点数)", fontsize=7)
    fig.tight_layout()
    p = OUT / "fig_new_validation_zoom.png"
    fig.savefig(p, bbox_inches="tight")
    print("saved", p)


if __name__ == "__main__":
    main()
