"""GM9 海域窗口最终结果图：KF 场 + 5–20 km 带 + GM9 断层。"""
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

FONT = Path(r"D:\gravity\fonts\SourceHanSansSC-Regular.otf")
if FONT.exists():
    fm.fontManager.addfont(str(FONT))
plt.rcParams["font.sans-serif"] = ["Source Han Sans SC", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).parent.parent / "output_real" / "three_parts"
LON, LAT = (170.3, 172.1), (-46.4, -45.0)


def main():
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
    b = bandpass(g, dx, dy, 5e3, 20e3)
    mask = filter_lineaments(lineament_mask(b, dx, dy, pct=70) & (bathy < 0),
                             min_area=20, min_elong=1.8)
    print("lineaments", mask.sum())

    d = json.loads((OUT / "nz_gm9_offshore.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(9.2, 7.0), dpi=200)
    v = np.percentile(np.abs(g), 99)
    ax.imshow(g, cmap="Greys", origin="lower", vmin=-v, vmax=v, alpha=0.55,
              interpolation="nearest", aspect="auto",
              extent=[lon[0], lon[-1], lat[0], lat[-1]])
    for f in d["features"]:
        geo = f.get("geometry") or {}
        if geo.get("type") != "LineString":
            continue
        a = np.asarray(geo["coordinates"], float)
        ax.plot(a[:, 0], a[:, 1], color="#C0392B", lw=1.5, alpha=0.95)
    ys, xs = np.where(mask)
    ax.scatter(lon[xs], lat[ys], s=0.6, c="#1F4E79", alpha=0.6, linewidths=0)
    ax.plot([], [], color="#C0392B", lw=1.5, label="GM9 海域断层")
    ax.scatter([], [], s=8, c="#1F4E79", label="KF 线性体（5–20 km）")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.set_xlim(lon[0], lon[-1]); ax.set_ylim(lat[0], lat[-1])
    ax.set_xlabel("经度 (°E)"); ax.set_ylabel("纬度 (°N)")
    ax.set_title("KF 线性体与 GM9 海域断层（5 km 覆盖 56.2%，随机 7.8%）",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = OUT / "fig_gm9_faults.png"
    fig.savefig(out, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
