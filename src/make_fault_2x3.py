"""断层页 2×3：场、带通、THG、线性体、覆盖率、距离分布。"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))

from run_pipeline import load_ssv_region
from three_method_compare import load_bathy_15s
from utils import cell_centers
from aniso_kf_test import build, LON, LAT
from scale_decomposition import bandpass, thg, lineament_mask, filter_lineaments
from fault_validation import densify, fault_match

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
RED = "#C0392B"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 8})


def main():
    g_lr, _, _, _ = load_ssv_region(LON, LAT)
    ny, nx = g_lr.shape; up = 4
    lon = cell_centers(LON[0], LON[1], nx * up)
    lat = cell_centers(LAT[0], LAT[1], ny * up)
    bathy = load_bathy_15s(LON, LAT, lon, lat)
    g = build(g_lr, bathy, lon, lat, w_keep=1.0)
    dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = float(lat[1] - lat[0]) * 111320
    b = bandpass(g, dx, dy, 5e3, 20e3)
    t = thg(b, dx, dy) * 1000
    mask = filter_lineaments(lineament_mask(b, dx, dy, pct=75), 20, 1.8)

    lines = []
    for f in json.load(open(ROOT / "output_real/three_parts/nz_gm9_offshore.json",
                            encoding="utf-8"))["features"]:
        geo = f.get("geometry") or {}
        if geo.get("type") != "LineString":
            continue
        a = np.asarray(geo["coordinates"], float)
        if ((a[:, 0] >= LON[0]) & (a[:, 0] <= LON[1]) &
                (a[:, 1] >= LAT[0]) & (a[:, 1] <= LAT[1])).any():
            lines.append(a)
    idx = np.arange(len(lines)); np.random.default_rng(7).shuffle(idx)
    tx, ty = densify([lines[i] for i in idx[len(lines) // 2:]], 0.5)
    st = fault_match(mask, tx, ty, lon, lat, radius_km=5)
    print("coverage", st["recall"] * 100)

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 5.6), dpi=220)
    k = 30
    ext = [lon[k], lon[-k], lat[k], lat[-k]]
    v = np.percentile(np.abs(g[k:-k, k:-k]), 99)
    axes[0, 0].imshow(g[k:-k, k:-k], cmap="RdBu_r", origin="lower", vmin=-v, vmax=v,
                      extent=ext, aspect="auto")
    axes[0, 0].set_title("(a) 相干保持 KF 15″ 场", fontsize=9)
    v = np.percentile(np.abs(b[k:-k, k:-k]), 99)
    axes[0, 1].imshow(b[k:-k, k:-k], cmap="RdBu_r", origin="lower", vmin=-v, vmax=v,
                      extent=ext, aspect="auto")
    axes[0, 1].set_title("(b) 5–20 km 带通", fontsize=9)
    vt = np.percentile(t[k:-k, k:-k], 99)
    axes[0, 2].imshow(t[k:-k, k:-k], cmap="inferno", origin="lower", vmin=0, vmax=vt,
                      extent=ext, aspect="auto")
    axes[0, 2].contour(lon[k:-k], lat[k:-k], mask[k:-k, k:-k], levels=[0.5],
                       colors="cyan", linewidths=0.4)
    axes[0, 2].set_title("(c) THG + 脊线", fontsize=9)

    depth = np.where(bathy < 0, bathy, 0)
    rgb = LightSource(azdeg=315, altdeg=45).shade(depth, cmap=plt.cm.Blues_r,
                                                 vmin=-5000, vmax=0, blend_mode="soft")
    axes[1, 0].imshow(rgb[k:-k, k:-k], origin="lower", extent=ext, aspect="auto")
    for a in lines:
        axes[1, 0].plot(a[:, 0], a[:, 1], color=RED, lw=0.9)
    ys, xs = np.where(mask)
    fp = np.column_stack(((tx - lon[0]) * 111320 * np.cos(np.radians(lat.mean())),
                          (ty - lat[0]) * 111320))
    pd = np.column_stack(((lon[xs] - lon[0]) * 111320 * np.cos(np.radians(lat.mean())),
                          (lat[ys] - lat[0]) * 111320))
    d_ft = cKDTree(fp).query(pd)[0] / 1000
    near = d_ft <= 5
    axes[1, 0].scatter(lon[xs][near], lat[ys][near], s=1.6, c="#2ECC71", linewidths=0)
    axes[1, 0].scatter(lon[xs][~near], lat[ys][~near], s=1.6, c="#F39C12", linewidths=0)
    axes[1, 0].set_title("(d) 线性体与 GM9（绿≤5 km）", fontsize=8.5)

    radii = np.arange(1, 21)
    tree_mask = cKDTree(pd)
    d_tm = tree_mask.query(fp)[0] / 1000
    cov = [(d_tm <= r).mean() * 100 for r in radii]
    axes[1, 1].plot(radii, cov, "o-", color="#0B2545", ms=3)
    axes[1, 1].axvline(5, color=RED, ls="--", lw=1)
    axes[1, 1].set_xlabel("缓冲半径 (km)", fontsize=8)
    axes[1, 1].set_ylabel("覆盖率 (%)", fontsize=8)
    axes[1, 1].set_title("(e) 覆盖率—缓冲半径", fontsize=9)
    axes[1, 1].grid(alpha=0.25); axes[1, 1].tick_params(labelsize=7)

    axes[1, 2].hist(d_ft[d_ft <= 30], bins=np.arange(0, 30, 1.5),
                    color="#0B2545", edgecolor="white")
    axes[1, 2].axvline(5, color=RED, ls="--", lw=1)
    axes[1, 2].set_xlabel("到最近断层的距离 (km)", fontsize=8)
    axes[1, 2].set_ylabel("线性体像元数", fontsize=8)
    axes[1, 2].set_title("(f) 距离分布", fontsize=9)
    axes[1, 2].grid(axis="y", alpha=0.25); axes[1, 2].tick_params(labelsize=7)

    for ax in axes.ravel()[:4]:
        ax.set_xlabel("经度 (°E)", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0, 0].set_ylabel("纬度 (°N)", fontsize=7)
    axes[1, 0].set_ylabel("纬度 (°N)", fontsize=7)
    fig.tight_layout()
    p = OUT / "fig_new_fault_2x3.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    main()
