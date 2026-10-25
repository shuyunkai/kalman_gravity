"""各向异性 KF 的 GM9 断层四联图。"""
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
OUT = ROOT / "output_real" / "three_parts" / "newdeck"
RED = "#C0392B"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False})


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
    test = [lines[i] for i in idx[len(lines) // 2:]]
    tx, ty = densify(test, 0.5)
    st = fault_match(mask, tx, ty, lon, lat, radius_km=5)
    print("coverage", st["recall"] * 100, "precision", st["precision"] * 100)

    fig, axes = plt.subplots(1, 4, figsize=(14.6, 3.7), dpi=220)
    k = 30
    ext = [lon[k], lon[-k], lat[k], lat[-k]]
    v = np.percentile(np.abs(g[k:-k, k:-k]), 99)
    axes[0].imshow(g[k:-k, k:-k], cmap="RdBu_r", origin="lower", vmin=-v, vmax=v,
                   extent=ext, aspect="auto")
    axes[0].set_title("(a) 各向异性 KF 15″ 场", fontsize=9)
    v = np.percentile(np.abs(b[k:-k, k:-k]), 99)
    axes[1].imshow(b[k:-k, k:-k], cmap="RdBu_r", origin="lower", vmin=-v, vmax=v,
                   extent=ext, aspect="auto")
    axes[1].set_title("(b) 5–20 km 带通", fontsize=9)
    vth = np.percentile(t[k:-k, k:-k], 99)
    axes[2].imshow(t[k:-k, k:-k], cmap="inferno", origin="lower", vmin=0, vmax=vth,
                   extent=ext, aspect="auto")
    axes[2].contour(lon[k:-k], lat[k:-k], mask[k:-k, k:-k], levels=[0.5],
                    colors="cyan", linewidths=0.45)
    axes[2].set_title("(c) THG + 提取脊线", fontsize=9)

    depth = np.where(bathy < 0, bathy, 0)
    rgb = LightSource(azdeg=315, altdeg=45).shade(depth, cmap=plt.cm.Blues_r,
                                                 vmin=-5000, vmax=0, blend_mode="soft")
    axes[3].imshow(rgb[k:-k, k:-k], origin="lower", extent=ext, aspect="auto")
    for a in lines:
        axes[3].plot(a[:, 0], a[:, 1], color=RED, lw=1.0)
    ys, xs = np.where(mask)
    fp = np.column_stack(((tx - lon[0]) * 111320 * np.cos(np.radians(lat.mean())),
                          (ty - lat[0]) * 111320))
    pd = np.column_stack(((lon[xs] - lon[0]) * 111320 * np.cos(np.radians(lat.mean())),
                          (lat[ys] - lat[0]) * 111320))
    dist = cKDTree(fp).query(pd)[0] / 1000
    near = dist <= 5
    axes[3].scatter(lon[xs][near], lat[ys][near], s=2.0, c="#2ECC71", linewidths=0)
    axes[3].scatter(lon[xs][~near], lat[ys][~near], s=2.0, c="#F39C12", linewidths=0)
    axes[3].set_title("(d) 线性体（绿≤5 km）与 GM9", fontsize=8)
    for ax in axes:
        ax.set_xlabel("经度 (°E)", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("纬度 (°N)", fontsize=7)
    fig.tight_layout()
    p = OUT / "fig_new_fault_aniso.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    main()
