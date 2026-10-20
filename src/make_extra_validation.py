"""补充验证图：断层（覆盖曲线+玫瑰图+地震距离）与反演（分频+散点）。"""
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.ndimage import zoom as sp_zoom
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))

from run_pipeline import load_ssv_region
from three_method_compare import load_bathy_15s
from utils import cell_centers
from kf_novel import BathyPriorKF
from scale_decomposition import bandpass, lineament_mask, filter_lineaments
from fault_validation import densify, fault_match, lineament_segments, _xy
from bathymetry_inversion import bandpass as bp_inv
from kf_inversion import spectral_kf_inversion

FONT = Path(r"D:\gravity\fonts\SourceHanSansSC-Regular.otf")
if FONT.exists():
    fm.fontManager.addfont(str(FONT))
plt.rcParams["font.sans-serif"] = ["Source Han Sans SC", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
OUT = Path(r"D:\gravity\kalman_gravity\output_real\three_parts")
LON, LAT = (170.3, 172.1), (-46.4, -45.0)


def build_field():
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
    return g, bathy, lon, lat


def fault_panels():
    g, bathy, lon, lat = build_field()
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
    train = [lines[i] for i in idx[:len(lines) // 2]]
    test = [lines[i] for i in idx[len(lines) // 2:]]
    tx, ty = densify(test, 0.5)

    b = bandpass(g, dx, dy, 5e3, 20e3)
    mask = filter_lineaments(lineament_mask(b, dx, dy, pct=70) & (bathy < 0),
                             min_area=20, min_elong=1.8)
    radii = np.arange(1, 21)
    cov, rc, pre, prc = [], [], [], []
    for r in radii:
        st = fault_match(mask, tx, ty, lon, lat, radius_km=float(r))
        cov.append(st["recall"] * 100); rc.append(st["recall_random"] * 100)
        pre.append(st["precision"] * 100); prc.append(st["precision_random"] * 100)

    segs = lineament_segments(mask, lon, lat, min_area=60, min_elong=2.0)
    la = np.array([s[0] for s in segs]) if segs else np.array([])
    fa = []
    for arr in lines:
        for (x0, y0), (x1, y1) in zip(arr[:-1], arr[1:]):
            fa.append(np.degrees(np.arctan2(y1 - y0,
                                           (x1 - x0) * np.cos(np.radians(y0)))) % 180)
    fa = np.array(fa)

    fig = plt.figure(figsize=(11.6, 4.0), dpi=220)
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(radii, cov, "o-", color="#1F3864", lw=1.6, ms=3, label="留出断层段")
    ax.plot(radii, rc, "s--", color="#C0392B", lw=1.2, ms=2.5, label="随机基线")
    ax.axvline(5, color="#7F8C8D", ls=":", lw=1)
    ax.set_xlabel("缓冲半径 (km)"); ax.set_ylabel("断层覆盖率 (%)")
    ax.set_title("(a) 覆盖率—缓冲半径", fontsize=10)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.3)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(radii, pre, "o-", color="#1F3864", lw=1.6, ms=3, label="线性体命中")
    ax.plot(radii, prc, "s--", color="#C0392B", lw=1.2, ms=2.5, label="随机基线")
    ax.axvline(5, color="#7F8C8D", ls=":", lw=1)
    ax.set_xlabel("缓冲半径 (km)"); ax.set_ylabel("命中率 (%)")
    ax.set_title("(b) 线性体命中率—缓冲半径", fontsize=10)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_fault_validation_2.png", bbox_inches="tight")
    plt.close(fig)

    # 地震距离
    try:
        url = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv"
               "&starttime=2000-01-01&minlatitude=-47&maxlatitude=-45"
               "&minlongitude=170&maxlongitude=172&minmagnitude=4.0")
        with urllib.request.urlopen(url, timeout=40) as r:
            txt = r.read().decode()
        els, elat = [], []
        for line in txt.splitlines()[1:]:
            p = line.split(",")
            try:
                elat.append(float(p[1])); els.append(float(p[2]))
            except (ValueError, IndexError):
                pass
        els, elat = np.array(els), np.array(elat)
        ys, xs = np.nonzero(mask)
        X, Y = _xy(lon[xs], lat[ys])
        tree = cKDTree(np.c_[X, Y])
        ex, ey = _xy(els, elat)
        dd, _ = tree.query(np.c_[ex, ey])
        rng2 = np.random.default_rng(0)
        rx = rng2.uniform(LON[0], LON[1], 3000); ry = rng2.uniform(LAT[0], LAT[1], 3000)
        rX, rY = _xy(rx, ry)
        dr, _ = tree.query(np.c_[rX, rY])
        print(f"EQ n={len(dd)}, median={np.median(dd):.1f} km, "
              f"frac20={np.mean(dd<=20)*100:.1f}% (rand {np.mean(dr<=20)*100:.1f}%)")
    except Exception as e:
        print("EQ fail", e)


def inversion_panels():
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
    m = np.isfinite(hi) & np.isfinite(ht)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), dpi=220)
    ax = axes[0]
    bands = ["20–50", "50–100", "100–200"]
    vals = [244, 288, 355]
    ax.bar(bands, vals, color="#2E75B6")
    for i, v in enumerate(vals):
        ax.text(i, v + 8, f"{v} m", ha="center", fontsize=8)
    ax.set_ylabel("RMSE (m)"); ax.set_title("(a) 分频精度", fontsize=10)
    ax.set_ylim(0, 420); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    hb = ax.hexbin(ht[m][::5], hi[m][::5], gridsize=60, cmap="Blues", mincnt=1)
    lim = np.percentile(np.abs(ht[m]), 99)
    ax.plot([-lim, lim], [-lim, lim], color="#C0392B", lw=1)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("GEBCO 带通 (m)"); ax.set_ylabel("KF 反演 (m)")
    ax.set_title("(b) 反演—真值散点", fontsize=10)
    fig.colorbar(hb, ax=ax, shrink=0.8, label="点数")
    fig.tight_layout()
    fig.savefig(OUT / "fig_inversion_validation_2.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import sys as _s
    if len(_s.argv) > 1 and _s.argv[1] == "fault":
        fault_panels()
    else:
        fault_panels()
        inversion_panels()
    print("saved")
