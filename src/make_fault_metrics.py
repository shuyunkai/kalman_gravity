"""断层页：上方总指标对比，下方场—带通—THG 过程图。"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.ndimage import zoom as sp_zoom
from scipy.interpolate import RegularGridInterpolator
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


def sdust_prior(lon, lat):
    ds = xr.open_dataset(ROOT / "data/SDUST2022/SDUST2022GRA.nc")
    sub = ds["ga"].sel(lon=slice(LON[0], LON[1]), lat=slice(LAT[0], LAT[1])).values
    glon = np.asarray(ds.lon.sel(lon=slice(LON[0], LON[1])))
    glat = np.asarray(ds.lat.sel(lat=slice(LAT[0], LAT[1])))
    ds.close()
    if glat[0] > glat[-1]:
        sub = sub[::-1]; glat = glat[::-1]
    f = RegularGridInterpolator((glat, glon), sub, bounds_error=False, fill_value=np.nan)
    LLON, LLAT = np.meshgrid(lon, lat)
    z = f(np.column_stack([LLAT.ravel(), LLON.ravel()])).reshape(LLAT.shape)
    return np.where(np.isfinite(z), z, np.nanmedian(z))


def main():
    g_lr, _, _, _ = load_ssv_region(LON, LAT)
    ny, nx = g_lr.shape; up = 4
    lon = cell_centers(LON[0], LON[1], nx * up)
    lat = cell_centers(LAT[0], LAT[1], ny * up)
    bathy = load_bathy_15s(LON, LAT, lon, lat)
    g = build(g_lr, bathy, lon, lat, w_keep=1.0, ssv_mode="obs_only", p0_obs=3.0, prior_long=sdust_prior(lon, lat))
    g1 = sp_zoom(g_lr, (up, up), order=1)
    dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = float(lat[1] - lat[0]) * 111320
    b = bandpass(g, dx, dy, 5e3, 20e3)
    t = thg(b, dx, dy) * 1000
    mask = filter_lineaments(lineament_mask(b, dx, dy, pct=60), 20, 1.8)
    b1 = bandpass(g1, dx, dy, 5e3, 20e3)
    mask1 = filter_lineaments(lineament_mask(b1, dx, dy, pct=60), 20, 1.8)

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
    tx, ty = densify([lines[i] for i in idx[len(lines)//2:]], 0.5)
    st = fault_match(mask, tx, ty, lon, lat, radius_km=5)
    st1 = fault_match(mask1, tx, ty, lon, lat, radius_km=5)
    cov = st["recall"] * 100; prec = st["precision"] * 100
    cov1 = st1["recall"] * 100; prec1 = st1["precision"] * 100

    fig, axes = plt.subplots(2, 3, figsize=(13.4, 5.4), dpi=220)

    ax = axes[0, 0]
    labels = ["5 km 缓冲覆盖率", "5 km 缓冲命中率"]
    std = [cov1, prec1]
    ckf = [cov, prec]
    y = np.arange(2); h = 0.30
    ax.barh(y+h/2, std, h, color="#90A4AE", label="SSV33.1（1′）")
    ax.barh(y-h/2, ckf, h, color="#0B2545", label="相干保持 KF")
    for i in range(2):
        ax.text(std[i]+1, y[i]+h/2, f"{std[i]:.1f}%", va="center", fontsize=7)
        ax.text(ckf[i]+1, y[i]-h/2, f"{ckf[i]:.1f}%", va="center", fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 100); ax.set_xlabel("百分比 (%)", fontsize=8)
    ax.set_title("(a) SSV33.1 与相干保持 KF", fontsize=10)
    ax.legend(fontsize=7, frameon=False); ax.tick_params(labelsize=7)
    ax.grid(axis="x", alpha=0.2)

    ax = axes[0, 1]
    ys, xs = np.where(mask)
    fp = np.column_stack(((tx - lon[0]) * 111320 * np.cos(np.radians(lat.mean())),
                          (ty - lat[0]) * 111320))
    pd = np.column_stack(((lon[xs] - lon[0]) * 111320 * np.cos(np.radians(lat.mean())),
                          (lat[ys] - lat[0]) * 111320))
    ys1, xs1 = np.where(mask1)
    pd1 = np.column_stack(((lon[xs1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean())),
                           (lat[ys1] - lat[0]) * 111320))
    d_tm = cKDTree(pd).query(fp)[0] / 1000
    d_tm1 = cKDTree(pd1).query(fp)[0] / 1000
    radii = np.arange(1, 21)
    cov_curve = [(d_tm <= r).mean() * 100 for r in radii]
    cov_curve1 = [(d_tm1 <= r).mean() * 100 for r in radii]
    ax.plot(radii, cov_curve1, "s--", color="#90A4AE", ms=2.5, label="SSV33.1")
    ax.plot(radii, cov_curve, "o-", color="#0B2545", ms=3, label="相干保持 KF")
    ax.axvline(5, color=RED, ls="--", lw=1)
    ax.set_xlabel("缓冲半径 (km)", fontsize=8); ax.set_ylabel("覆盖率 (%)", fontsize=8)
    ax.set_title("(b) 覆盖率—缓冲半径", fontsize=10)
    ax.legend(fontsize=6, frameon=False); ax.grid(alpha=0.25); ax.tick_params(labelsize=7)

    ax = axes[0, 2]
    d_ft = cKDTree(fp).query(pd)[0] / 1000
    d_ft1 = cKDTree(fp).query(pd1)[0] / 1000
    ax.hist(d_ft1[d_ft1 <= 30], bins=np.arange(0, 30, 1.5), color="#90A4AE",
            edgecolor="white", alpha=0.7, label="SSV33.1")
    ax.hist(d_ft[d_ft <= 30], bins=np.arange(0, 30, 1.5), color="#0B2545",
            edgecolor="white", alpha=0.6, label="相干保持 KF")
    ax.axvline(5, color=RED, ls="--", lw=1)
    ax.set_xlabel("到最近断层的距离 (km)", fontsize=8)
    ax.set_ylabel("线性体像元数", fontsize=8)
    ax.set_title("(c) 距离分布", fontsize=10)
    ax.legend(fontsize=6, frameon=False); ax.grid(axis="y", alpha=0.25)
    ax.tick_params(labelsize=7)

    k = 30
    ext = [lon[k], lon[-k], lat[k], lat[-k]]
    ax = axes[1, 0]
    v = np.percentile(np.abs(g1[k:-k, k:-k]), 99)
    im = ax.imshow(g1[k:-k, k:-k], cmap="RdBu_r", origin="lower", vmin=-v, vmax=v,
                   extent=ext, aspect="auto")
    ax.set_title("(d) SSV33.1（1′，15″ 格网）", fontsize=10)
    ax.set_xlabel("经度 (°E)", fontsize=7); ax.set_ylabel("纬度 (°N)", fontsize=7)
    ax.tick_params(labelsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("mGal", fontsize=6)

    ax = axes[1, 1]
    v = np.percentile(np.abs(g[k:-k, k:-k]), 99)
    im = ax.imshow(g[k:-k, k:-k], cmap="RdBu_r", origin="lower", vmin=-v, vmax=v,
                   extent=ext, aspect="auto")
    ax.set_title("(e) 相干保持 KF 场", fontsize=10)
    ax.set_xlabel("经度 (°E)", fontsize=7); ax.set_ylabel("纬度 (°N)", fontsize=7); ax.tick_params(labelsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("mGal", fontsize=6)

    ax = axes[1, 2]
    vt = np.percentile(t[k:-k, k:-k], 99)
    im = ax.imshow(t[k:-k, k:-k], cmap="inferno", origin="lower", vmin=0, vmax=vt,
                   extent=ext, aspect="auto")
    ax.contour(lon[k:-k], lat[k:-k], mask[k:-k, k:-k], levels=[0.5],
               colors="cyan", linewidths=0.4)
    fault = np.array([[170.70, -46.20], [170.72, -45.95], [170.78, -45.75],
                      [170.90, -45.60], [171.05, -45.50], [171.25, -45.40],
                      [171.45, -45.32], [171.60, -45.25]])
    ax.plot(fault[:, 0], fault[:, 1], color="red", lw=1.4, ls="--")
    ax.annotate("大断裂：宽缓边界\nTHG 未达阈值", xy=(171.00, -45.55),
                xytext=(170.48, -45.38), fontsize=7.5, color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
    ax.set_title("(f) THG、脊线（青）与大断裂（红）", fontsize=10)
    ax.set_xlabel("经度 (°E)", fontsize=7); ax.set_ylabel("纬度 (°N)", fontsize=7); ax.tick_params(labelsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("THG (Eötvös)", fontsize=6)

    fig.tight_layout()
    p = OUT / "fig_new_fault_metrics.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("saved", p, "cov", cov, "prec", prec)


if __name__ == "__main__":
    main()
