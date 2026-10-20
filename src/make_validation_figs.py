"""验证图：断层缓冲曲线 + 反演分频/交叉验证。"""
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
from fault_validation import densify, fault_match

FONT = Path(r"D:\gravity\fonts\SourceHanSansSC-Regular.otf")
if FONT.exists():
    fm.fontManager.addfont(str(FONT))
plt.rcParams["font.sans-serif"] = ["Source Han Sans SC", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
OUT = Path(r"D:\gravity\kalman_gravity\output_real\three_parts")
LON, LAT = (170.3, 172.1), (-46.4, -45.0)


def fault_curve():
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
    radii = np.arange(1, 21)
    cov, pre, rc, rp = [], [], [], []
    for r in radii:
        st = fault_match(mask, tx, ty, lon, lat, radius_km=float(r))
        cov.append(st["recall"] * 100); rc.append(st["recall_random"] * 100)

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=220)
    ax.plot(radii, cov, "o-", color="#1F3864", lw=1.6, ms=3, label="留出断层段")
    ax.plot(radii, rc, "s--", color="#C0392B", lw=1.2, ms=2.5, label="随机基线")
    ax.axvline(5, color="#7F8C8D", ls=":", lw=1)
    ax.set_xlabel("缓冲半径 (km)"); ax.set_ylabel("断层覆盖率 (%)")
    ax.set_title("GM9 海域：覆盖率随缓冲半径的变化", fontsize=10.5)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_fault_validation_curve.png", bbox_inches="tight")
    plt.close(fig)


def inversion_bars():
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9), dpi=220)
    ax = axes[0]
    bands = ["20–50", "50–100", "100–200"]
    vals = [244, 288, 355]
    ax.bar(bands, vals, color="#2E75B6")
    for i, v in enumerate(vals):
        ax.text(i, v + 8, f"{v} m", ha="center", fontsize=8)
    ax.set_ylabel("RMSE (m)"); ax.set_title("分频段反演精度", fontsize=10)
    ax.set_ylim(0, 420); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    names = ["KF\nvs GEBCO", "Parker\nvs GEBCO", "KF\nvs GMRT"]
    vals = [651, 651, 648]
    ax.bar(names, vals, color=["#1F3864", "#7F8C8D", "#C0392B"])
    for i, v in enumerate(vals):
        ax.text(i, v + 6, f"{v} m", ha="center", fontsize=8)
    ax.set_ylabel("RMSE (m)"); ax.set_title("全带与独立产品交叉验证", fontsize=10)
    ax.set_ylim(0, 760); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_inversion_validation.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fault_curve()
    inversion_bars()
    print("saved")
