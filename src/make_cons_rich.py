"""缺点页 2×2：1′ 场、相干保持 KF 场、差值、分频幅值。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from scale_decomposition import bandpass

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 10})


def main():
    d = np.load(OUT / "validation_G_fields.npz")
    lon, lat = d["lon"], d["lat"]
    ssv, ckf = d["ssv"], d["ckf"]
    ext = [lon[0], lon[-1], lat[0], lat[-1]]
    k = 4
    sub = (slice(k, -k), slice(k, -k))
    v = np.nanpercentile(np.abs(ckf[sub]), 98)
    dx = (lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = (lat[1] - lat[0]) * 111320

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 5.2), dpi=220)
    for ax, name, fld in ((axes[0, 0], "(a) 1′ 产品", ssv),
                          (axes[0, 1], "(b) 相干保持 KF", ckf)):
        im = ax.imshow(fld[sub], cmap="RdBu_r", origin="lower",
                       vmin=-v, vmax=v, extent=ext, aspect="auto")
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("经度 (°E)", fontsize=10)
        ax.set_ylabel("纬度 (°N)", fontsize=10)
        ax.tick_params(labelsize=9)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("mGal", fontsize=9); cb.ax.tick_params(labelsize=8)
    ax = axes[1, 0]
    diff = ckf - ssv
    vd = np.nanpercentile(np.abs(diff[sub]), 98)
    im = ax.imshow(diff[sub], cmap="PuOr_r", origin="lower", vmin=-vd, vmax=vd,
                   extent=ext, aspect="auto")
    ax.set_title("(c) 相干保持 KF − 1′ 差值", fontsize=12)
    ax.set_xlabel("经度 (°E)", fontsize=10)
    ax.set_ylabel("纬度 (°N)", fontsize=10)
    ax.tick_params(labelsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("mGal", fontsize=9); cb.ax.tick_params(labelsize=8)

    ax = axes[1, 1]
    bands = [(5e3, 10e3, "5–10"), (10e3, 20e3, "10–20"),
             (20e3, 50e3, "20–50")]
    r1 = [float(np.std(bandpass(ssv, dx, dy, a, b))) for a, b, _ in bands]
    r2 = [float(np.std(bandpass(ckf, dx, dy, a, b))) for a, b, _ in bands]
    x = np.arange(3); w = 0.34
    ax.bar(x - w/2, r1, w, color="#D6E2F0", label="1′")
    ax.bar(x + w/2, r2, w, color="#0B2545", label="相干保持 KF")
    for i in range(3):
        ax.text(x[i]-w/2, r1[i]+0.1, f"{r1[i]:.1f}", ha="center", fontsize=8)
        ax.text(x[i]+w/2, r2[i]+0.1, f"{r2[i]:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([b[2] for b in bands], fontsize=10)
    ax.set_ylabel("幅值 std (mGal)", fontsize=10)
    ax.set_title("(d) 分频幅值", fontsize=12)
    ax.legend(fontsize=9, frameon=False)
    ax.tick_params(labelsize=9); ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    p = OUT / "fig_new_cons_rich.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    main()
