"""产品页 2×2：全球场、区域 1′、产品卡、三区柱状。"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 10})


def main():
    d = np.load(OUT / "validation_G_fields.npz")
    lon, lat, ssv = d["lon"], d["lat"], d["ssv"]
    ext = [lon[0], lon[-1], lat[0], lat[-1]]
    k = 4
    sub = (slice(k, -k), slice(k, -k))

    g = xr.open_dataset(ROOT / "data/icgem/grid/EGM2008_gravity_anomaly.nc")
    gv = g["gravity_anomaly"].values[::20, ::20]
    glon = g["lon"].values[::20]; glat = g["lat"].values[::20]

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 5.2), dpi=220)
    ax = axes[0, 0]
    im = ax.imshow(gv, cmap="RdBu_r", origin="upper",
                   extent=[glon[0], glon[-1], glat[-1], glat[0]],
                   vmin=-100, vmax=100, aspect="auto")
    ax.set_title("(a) 全球重力异常（EGM2008）", fontsize=12)
    ax.set_xlabel("经度 (°E)", fontsize=10); ax.set_ylabel("纬度 (°N)", fontsize=10)
    ax.tick_params(labelsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("mGal", fontsize=9); cb.ax.tick_params(labelsize=8)

    ax = axes[0, 1]
    v = np.nanpercentile(np.abs(ssv[sub]), 98)
    im = ax.imshow(ssv[sub], cmap="RdBu_r", origin="lower", vmin=-v, vmax=v,
                   extent=ext, aspect="auto")
    ax.set_title("(b) 区域 1′ 产品（SSV33.1）", fontsize=12)
    ax.set_xlabel("经度 (°E)", fontsize=10); ax.set_ylabel("纬度 (°N)", fontsize=10)
    ax.tick_params(labelsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("mGal", fontsize=9); cb.ax.tick_params(labelsize=8)

    ax = axes[1, 0]; ax.axis("off")
    items = ["SSV33.1", "DTU21GRA", "SDUST2022GRA", "NSOAS24"]
    subs = ["1′ / 全球", "1′ / 全球", "1′ / 中国近海", "1′ / 全球"]
    for i, (name, sub) in enumerate(zip(items, subs)):
        y = 0.78 - i * 0.22
        ax.add_patch(plt.Rectangle((0.02, y), 0.96, 0.17, fc="#EEF3F9",
                                   ec="#0B2545", lw=1))
        ax.text(0.06, y + 0.085, name, va="center", fontsize=11, color="#0B2545")
        ax.text(0.62, y + 0.085, sub, va="center", fontsize=9, color="#5B6573")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("(c) 常用 1′ 产品", fontsize=12)

    ax = axes[1, 1]
    res = json.loads((OUT / "regions_coherence.json").read_text(encoding="utf-8"))
    names = ["南海-琉球", "菲律宾海", "日本"]
    keys = ["D_Nankai", "F_PhilSea", "G_Japan"]
    x = np.arange(3); w = 0.34
    ax.bar(x - w/2, [res[k]["ssv"] for k in keys], w, color="#D6E2F0", label="1′")
    ax.bar(x + w/2, [res[k]["ckf"] for k in keys], w, color="#0B2545", label="CKF")
    for i, k in enumerate(keys):
        ax.text(x[i] - w/2, res[k]["ssv"] + 0.1, f"{res[k]['ssv']:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("RMSE (mGal)", fontsize=10)
    ax.set_title("(d) 三区留出精度", fontsize=12)
    ax.legend(fontsize=9, frameon=False); ax.tick_params(labelsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    p = OUT / "fig_new_products_rich.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    main()
