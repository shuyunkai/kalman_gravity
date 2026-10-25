"""总结页：三张 KPI + 最终场 + 差值图。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 11})


def main():
    d = np.load(OUT / "validation_G_fields.npz")
    lon, lat, ssv, ckf = d["lon"], d["lat"], d["ssv"], d["ckf"]
    ext = [lon[0], lon[-1], lat[0], lat[-1]]
    k = 4
    sub = (slice(k, -k), slice(k, -k))
    v = np.nanpercentile(np.abs(ckf[sub]), 98)

    fig = plt.figure(figsize=(13.4, 5.2), dpi=220)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 2.4], hspace=0.28, wspace=0.16)

    cards = [("15″", "分辨率", "1.85 km 网格"),
             ("0.2–8.7%", "精度改善", "相对 1′ 产品"),
             ("67.9%", "断裂覆盖", "5 km，GM9")]
    for i, (val, name, sub2) in enumerate(cards):
        ax = fig.add_subplot(gs[0, i]); ax.axis("off")
        ax.add_patch(plt.Rectangle((0.02, 0.05), 0.96, 0.90, fc="#EEF3F9",
                                   ec="#0B2545", lw=1.2,
                                   joinstyle="round", capstyle="round"))
        ax.text(0.5, 0.66, val, ha="center", fontsize=24, color="#C0392B",
                fontweight="bold")
        ax.text(0.5, 0.34, name, ha="center", fontsize=13, color="#0B2545",
                fontweight="bold")
        ax.text(0.5, 0.14, sub2, ha="center", fontsize=10, color="#5B6573")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax = fig.add_subplot(gs[1, :2])
    im = ax.imshow(ckf[sub], cmap="RdBu_r", origin="lower", vmin=-v, vmax=v,
                   extent=ext, aspect="auto")
    ax.set_title("(a) 相干保持 KF 15″ 重力场", fontsize=12)
    ax.set_xlabel("经度 (°E)", fontsize=10); ax.set_ylabel("纬度 (°N)", fontsize=10)
    ax.tick_params(labelsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("mGal", fontsize=9); cb.ax.tick_params(labelsize=8)

    ax = fig.add_subplot(gs[1, 2])
    diff = ckf - ssv
    vd = np.nanpercentile(np.abs(diff[sub]), 98)
    im = ax.imshow(diff[sub], cmap="PuOr_r", origin="lower", vmin=-vd, vmax=vd,
                   extent=ext, aspect="auto")
    ax.set_title("(b) 15″ − 1′", fontsize=12)
    ax.set_xlabel("经度 (°E)", fontsize=10); ax.set_ylabel("纬度 (°N)", fontsize=10)
    ax.tick_params(labelsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("mGal", fontsize=9); cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    p = OUT / "fig_new_summary_rich.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    main()
