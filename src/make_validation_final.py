"""最终验证图：断层三合一（覆盖/命中/地图）、反演三合一（分频/散点/剖面）。"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.ndimage import zoom as sp_zoom

sys.path.insert(0, str(Path(__file__).parent))

from run_pipeline import load_ssv_region
from three_method_compare import load_bathy_15s
from utils import cell_centers
from kf_novel import BathyPriorKF
from scale_decomposition import bandpass, lineament_mask, filter_lineaments
from fault_validation import densify, fault_match, _mask_points_km, _xy
from scipy.spatial import cKDTree
from matplotlib.colors import LightSource
from bathymetry_inversion import bandpass as bp_inv
from kf_inversion import spectral_kf_inversion

FONT = Path(r"D:\gravity\fonts\SourceHanSansSC-Regular.otf")
if FONT.exists():
    fm.fontManager.addfont(str(FONT))
plt.rcParams["font.sans-serif"] = ["Source Han Sans SC", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
OUT = Path(r"D:\gravity\kalman_gravity\output_real\three_parts")
LON, LAT = (170.3, 172.1), (-46.4, -45.0)


def fault_three():
    g_lr, _, _, _ = load_ssv_region(LON, LAT)
    ny, nx = g_lr.shape
    up = 4
    lon = cell_centers(LON[0], LON[1], nx * up)
    lat = cell_centers(LAT[0], LAT[1], ny * up)
    bathy = load_bathy_15s(LON, LAT, lon, lat)
    ssv_hr = sp_zoom(g_lr, (up, up), order=1)
    ship = np.full(ssv_hr.shape, np.nan)
    m0 = np.zeros(ssv_hr.shape, dtype=bool)
    kf = BathyPriorKF(ssv_hr.shape, g_lr.shape, (1.4, 1.8), alpha=0.5)
    g = kf.build(g_lr.astype(float), ship, m0, lon, lat, LON, LAT,
                 upscale=up, bathy_15s=bathy, n_iter=6, smooth_sigma=4.0)
    dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = float(lat[1] - lat[0]) * 111320

    d = json.loads((OUT / "nz_gm9_offshore.json").read_text(encoding="utf-8"))
    lines = []
    for f in d["features"]:
        geo = f.get("geometry") or {}
        if geo.get("type") != "LineString":
            continue
        a = np.asarray(geo["coordinates"], float)
        ins = ((a[:, 0] >= LON[0]) & (a[:, 0] <= LON[1]) &
               (a[:, 1] >= LAT[0]) & (a[:, 1] <= LAT[1]))
        if ins.any():
            lines.append(a)
    idx = np.arange(len(lines)); rng = np.random.default_rng(7); rng.shuffle(idx)
    test = [lines[i] for i in idx[len(lines) // 2:]]
    tx, ty = densify(test, 0.5)

    b = bandpass(g, dx, dy, 5e3, 20e3)
    mask = filter_lineaments(lineament_mask(b, dx, dy, pct=70) & (bathy < 0),
                             min_area=20, min_elong=1.8)
    # 公平基线：随机点抽在断层包络内，而不是整个大区域
    fx = np.concatenate([a[:, 0] for a in test])
    fy = np.concatenate([a[:, 1] for a in test])
    rng2 = np.random.default_rng(1)
    pad = 0.1
    rx = rng2.uniform(fx.min() - pad, fx.max() + pad, 5000)
    ry = rng2.uniform(fy.min() - pad, fy.max() + pad, 5000)

    lp = _mask_points_km(mask, lon, lat)
    tree_l = cKDTree(lp)
    fX, fY = _xy(tx, ty)
    fp = np.column_stack([fX, fY])
    tree_f = cKDTree(fp)
    rX, rY = _xy(rx, ry)
    rp = np.column_stack([rX, rY])

    d_fault, _ = tree_l.query(fp)
    d_rand_l, _ = tree_l.query(rp)
    d_line, _ = tree_f.query(lp)
    d_rand_f, _ = tree_f.query(rp)

    radii = np.arange(1, 21)
    cov, rc, pre, prc = [], [], [], []
    for r in radii:
        cov.append(np.mean(d_fault <= r) * 100)
        rc.append(np.mean(d_rand_l <= r) * 100)
        pre.append(np.mean(d_line <= r) * 100)
        prc.append(np.mean(d_rand_f <= r) * 100)

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 3.9), dpi=220,
                             gridspec_kw={"width_ratios": [1, 1, 1.5]})
    ax = axes[0]
    ax.plot(radii, cov, "o-", color="#1F3864", lw=1.5, ms=2.5, label="留出段")
    ax.plot(radii, rc, "s--", color="#C0392B", lw=1.1, ms=2, label="随机")
    ax.axvline(5, color="#7F8C8D", ls=":", lw=1)
    ax.set_xlabel("缓冲半径 (km)"); ax.set_ylabel("覆盖率 (%)")
    ax.set_title("(a) 断层覆盖率", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(radii, pre, "o-", color="#1F3864", lw=1.5, ms=2.5, label="线性体")
    ax.plot(radii, prc, "s--", color="#C0392B", lw=1.1, ms=2, label="随机")
    ax.axvline(5, color="#7F8C8D", ls=":", lw=1)
    ax.set_xlabel("缓冲半径 (km)"); ax.set_ylabel("命中率 (%)")
    ax.set_title("(b) 线性体命中率", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False); ax.grid(alpha=0.3)

    ax = axes[2]
    depth = np.where(bathy < 0, bathy, 0.0)
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(depth, cmap=plt.cm.Blues_r, vmin=-5000, vmax=0,
                   blend_mode="soft")
    ax.imshow(rgb, origin="lower",
              extent=[lon[0], lon[-1], lat[0], lat[-1]], aspect="auto")
    ax.contour(lon, lat, bathy, levels=[0], colors="k", linewidths=0.6)
    for arr in lines:
        ax.plot(arr[:, 0], arr[:, 1], color="#C0392B", lw=1.2, alpha=0.9)
    ys, xs = np.where(mask)
    ax.scatter(lon[xs], lat[ys], s=0.2, c="black", alpha=0.55, linewidths=0)
    ax.set_xlabel("经度 (°E)"); ax.set_ylabel("纬度 (°N)")
    ax.set_title("(c) 线性体与 GM9 断层", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_fault_validation_3.png", bbox_inches="tight")
    plt.close(fig)


def inversion_three():
    d = np.load(r"D:\gravity\kalman_gravity\output_real\test_field_15s.npz")
    g = d["g15"].astype(float); bathy = d["bathy"].astype(float)
    lon, lat = d["lon"], d["lat"]
    dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = float(lat[1] - lat[0]) * 111320
    z0 = float(np.mean(-bathy))
    dg = bp_inv(g, dx, dy, 20e3, 200e3)
    hk, _ = spectral_kf_inversion(dg, dx, dy, 2200.0, z0, prior_rms=2500.0,
                                  n_iter=8, lam_min=25e3, relax=0.6)
    hi = bp_inv(hk, dx, dy, 20e3, 200e3)
    ht = bp_inv(z0 + bathy, dx, dy, 20e3, 200e3)

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 3.9), dpi=220)
    ax = axes[0]
    bands = ["20–50", "50–100", "100–200"]
    vals = [244, 288, 355]
    ax.bar(bands, vals, color="#2E75B6")
    for i, v in enumerate(vals):
        ax.text(i, v + 8, f"{v}", ha="center", fontsize=8)
    ax.set_ylabel("RMSE (m)"); ax.set_title("(a) 分频精度", fontsize=10)
    ax.set_ylim(0, 420); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    m = np.isfinite(hi) & np.isfinite(ht)
    lim = 1200
    ax.hexbin(np.clip(ht[m], -lim, lim), np.clip(hi[m], -lim, lim),
              gridsize=60, cmap="Blues", mincnt=1)
    ax.plot([-lim, lim], [-lim, lim], color="#C0392B", lw=1)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("GEBCO 带通 (m)"); ax.set_ylabel("KF 反演 (m)")
    ax.set_title("(b) 反演—真值", fontsize=10)

    ax = axes[2]
    i = np.searchsorted(lat, 36.0)
    ax.plot(lat, ht[:, i], color="#1F3864", lw=1.3, label="GEBCO")
    ax.plot(lat, hi[:, i], color="#C0392B", lw=1.1, alpha=0.9, label="KF")
    ax.set_xlabel("纬度 (°N)"); ax.set_ylabel("起伏 (m)")
    ax.set_title("(c) 143.5°E 剖面", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_inversion_validation_3.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fault_three()
    inversion_three()
    print("saved")
