"""标准 KF 问题页 2×2。"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})


def main():
    d = np.load(OUT / "validation_G_fields.npz")
    lon, lat, ssv, ckf = d["lon"], d["lat"], d["ssv"], d["ckf"]
    ext = [lon[0], lon[-1], lat[0], lat[-1]]
    k = 4; sub = (slice(k, -k), slice(k, -k))

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 5.2), dpi=220)
    ax = axes[0, 0]
    gy, gx = np.mgrid[0:9, 0:9]
    ax.scatter(gx, gy, s=45, c="#0B2545")
    ax.set_xlim(-0.5, 8.5); ax.set_ylim(-0.5, 8.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(a) 对角 P：格点被当作独立", fontsize=11)
    ax.set_aspect("equal")

    ax = axes[0, 1]
    ax.add_patch(plt.Rectangle((0.08, 0.08), 0.84, 0.84, fill=False,
                               ec="#C0392B", lw=1.8))
    for i in range(4):
        for j in range(4):
            ax.scatter([0.20 + i*0.20], [0.20 + j*0.20], s=30, c="#9FB3C8")
    ax.scatter([0.5], [0.5], s=70, c="#0B2545", zorder=5)
    ax.text(0.5, 0.02, "1′ 块内 16 个 15″ 像素，只给一个观测值", ha="center",
            fontsize=8, color="#C0392B")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("(b) 1′ 块平均 ≠ 点观测", fontsize=11)

    ax = axes[1, 0]
    diff = ckf - ssv
    vd = np.nanpercentile(np.abs(diff[sub]), 98)
    im = ax.imshow(diff[sub], cmap="PuOr_r", origin="lower", vmin=-vd, vmax=vd,
                   extent=ext, aspect="auto")
    ax.set_title("(c) 船测条带 / 块边界", fontsize=11)
    ax.set_xlabel("经度 (°E)", fontsize=9); ax.set_ylabel("纬度 (°N)", fontsize=9)
    ax.tick_params(labelsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("mGal", fontsize=8); cb.ax.tick_params(labelsize=7)

    ax = axes[1, 1]
    res = json.loads((OUT / "regions_coherence.json").read_text(encoding="utf-8"))
    names = ["南海-琉球", "菲律宾海", "日本"]
    keys = ["D_Nankai", "F_PhilSea", "G_Japan"]
    x = np.arange(3); w = 0.34
    ax.bar(x - w/2, [res[q]["ssv"] for q in keys], w, color="#D6E2F0", label="1′")
    ax.bar(x + w/2, [res[q]["ckf"] for q in keys], w, color="#0B2545", label="相干保持 KF")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("RMSE (mGal)", fontsize=9)
    ax.set_title("(d) 固定 R：不同海域误差差异", fontsize=11)
    ax.legend(fontsize=8, frameon=False); ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    p = OUT / "fig_new_stdkf_problems_2x2.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    main()
