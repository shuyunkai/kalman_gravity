"""科研风格图：三任务合图（真实场）+ 真实 KF 尺度分量。"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

sys.path.insert(0, str(Path(__file__).parent))

from scale_decomposition import bandpass, thg, lineament_mask, filter_lineaments
from fault_validation import load_gem_faults
from bathymetry_inversion import bandpass as bp_inv
from kf_inversion import spectral_kf_inversion

FONT = Path(r"D:\gravity\fonts\SourceHanSansSC-Regular.otf")
if FONT.exists():
    fm.fontManager.addfont(str(FONT))
plt.rcParams["font.sans-serif"] = ["Source Han Sans SC", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real" / "three_parts"
NPZ = ROOT / "output_real" / "test_field_15s.npz"


def three_tasks():
    d = np.load(NPZ)
    g = d["g15"].astype(float); bathy = d["bathy"].astype(float)
    lon, lat = d["lon"], d["lat"]
    dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = float(lat[1] - lat[0]) * 111320

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), dpi=220)
    ax = axes[0]
    v = np.percentile(np.abs(g), 99)
    im = ax.imshow(g, cmap="RdBu_r", origin="lower", vmin=-v, vmax=v,
                   extent=[lon[0], lon[-1], lat[0], lat[-1]], aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.85).set_label("mGal")
    ax.set_title("(a) 15″ 重力场（KF 重建）", fontsize=10)
    ax.set_xlabel("经度 (°E)"); ax.set_ylabel("纬度 (°N)")

    ax = axes[1]
    b = bandpass(g, dx, dy, 5e3, 20e3)
    c = 30
    vv = np.percentile(np.abs(b), 98)
    im = ax.imshow(b[c:-c, c:-c], cmap="RdBu_r", origin="lower", vmin=-vv, vmax=vv,
                   extent=[lon[c], lon[-c - 1], lat[c], lat[-c - 1]], aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.85).set_label("mGal")
    mask = filter_lineaments(lineament_mask(b, dx, dy, pct=70) & (bathy < 0),
                             min_area=20, min_elong=1.8)
    ys, xs = np.where(mask)
    keep = (xs > c) & (xs < len(lon) - c) & (ys > c) & (ys < len(lat) - c)
    ax.scatter(lon[xs[keep]], lat[ys[keep]], s=0.2, c="black", alpha=0.55,
               linewidths=0)
    ax.set_title("(b) 5–20 km 尺度带与线性体", fontsize=10)
    ax.set_xlabel("经度 (°E)"); ax.set_ylabel("纬度 (°N)")

    ax = axes[2]
    z0 = float(np.mean(-bathy))
    dg = bp_inv(g, dx, dy, 20e3, 200e3)
    hk, _ = spectral_kf_inversion(dg, dx, dy, 2200.0, z0, prior_rms=2500.0,
                                  n_iter=8, lam_min=25e3, relax=0.6)
    hi = bp_inv(hk, dx, dy, 20e3, 200e3)
    ht = bp_inv(z0 + bathy, dx, dy, 20e3, 200e3)
    i = np.searchsorted(lat, 36.0)
    ax.plot(lat, ht[:, i], color="#1F3864", lw=1.4, label="GEBCO 带通")
    ax.plot(lat, hi[:, i], color="#C0392B", lw=1.2, label="Parker 反演")
    ax.set_xlabel("纬度 (°N)"); ax.set_ylabel("相对起伏 (m)")
    ax.set_title("(c) 反演剖面（143.5°E）", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_three_tasks.png", bbox_inches="tight")
    plt.close(fig)


def real_scales():
    d = np.load(OUT / "multiscale_kf.npz")
    x1, x2, x3 = d["x1"], d["x2"], d["x3"]
    lon, lat = d["lon"], d["lat"]
    panels = [(x1, "$x_1$ 粗尺度"), (x2, "$x_2$ 中尺度"),
              (x3, "$x_3$ 细尺度"), (x1 + x2 + x3, "合成 $x$")]
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.7), dpi=220)
    for ax, (a, t) in zip(axes, panels):
        v = np.percentile(np.abs(a), 99)
        im = ax.imshow(a, cmap="RdBu_r", origin="lower", vmin=-v, vmax=v,
                       extent=[lon[0], lon[-1], lat[0], lat[-1]], aspect="auto")
        ax.set_title(t, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8).set_label("mGal", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_kf_scales_real.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    real_scales()
    three_tasks()
    print("saved")
