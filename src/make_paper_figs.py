"""按公开论文常见版式生成验证图与方法流程图。"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))

from bathymetry_inversion import bandpass as bp, parker_inversion
from kf_inversion import spectral_kf_inversion
from scale_decomposition import bandpass, lineament_mask, filter_lineaments, thg
from fault_validation import densify, fault_match, lineament_segments, _mask_points_km, _xy

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real" / "three_parts"
FONT = Path(r"D:\gravity\fonts\SourceHanSansSC-Regular.otf")
if FONT.exists():
    from matplotlib import font_manager as fm
    fm.fontManager.addfont(str(FONT))
plt.rcParams.update({
    "font.sans-serif": ["Source Han Sans SC", "Microsoft YaHei"],
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
})

NAVY = "#1F3864"; RED = "#C0392B"; GREY = "#5B6573"


def build_gm9():
    from scipy.ndimage import zoom as sp_zoom
    from run_pipeline import load_ssv_region
    from three_method_compare import load_bathy_15s
    from utils import cell_centers
    from kf_novel import BathyPriorKF

    LON, LAT = (170.3, 172.1), (-46.4, -45.0)
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
    g = kf.build(g_lr.astype(float), ship, m0, lon, lat, LON, LAT, upscale=up,
                 bathy_15s=bathy, n_iter=6, smooth_sigma=4.0)
    return g, bathy, lon, lat


def fault_figure():
    g, bathy, lon, lat = build_gm9()
    dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = float(lat[1] - lat[0]) * 111320
    b = bandpass(g, dx, dy, 5e3, 20e3)
    mask = filter_lineaments(lineament_mask(b, dx, dy, pct=70) & (bathy < 0), 20, 1.8)

    lines = []
    for f in json.loads((OUT / "nz_gm9_offshore.json").read_text(encoding="utf-8"))["features"]:
        geo = f.get("geometry") or {}
        if geo.get("type") != "LineString":
            continue
        a = np.asarray(geo["coordinates"], float)
        if ((a[:, 0] >= lon.min()) & (a[:, 0] <= lon.max()) &
                (a[:, 1] >= lat.min()) & (a[:, 1] <= lat.max())).any():
            lines.append(a)
    idx = np.arange(len(lines)); np.random.default_rng(7).shuffle(idx)
    test = [lines[i] for i in idx[len(lines) // 2:]]
    tx, ty = densify(test, 0.5)

    fig = plt.figure(figsize=(14.6, 3.9), dpi=220)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.55, 1, 0.9, 1.05], wspace=0.38)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2], projection="polar")
    ax3 = fig.add_subplot(gs[0, 3])
    # (a) map
    ax = ax0
    depth = np.where(bathy < 0, bathy, 0.0)
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(depth, cmap=plt.cm.Blues_r, vmin=-5000, vmax=0, blend_mode="soft")
    ax.imshow(rgb, origin="lower", extent=[lon[0], lon[-1], lat[0], lat[-1]], aspect="auto")
    for arr in lines:
        ax.plot(arr[:, 0], arr[:, 1], color=RED, lw=1.0, alpha=0.9)
    ys, xs = np.where(mask)
    ax.scatter(lon[xs], lat[ys], s=0.35, c="#FFD400", alpha=0.9, linewidths=0)
    ax.set_title("(a) 线性体与断层", fontsize=10)
    ax.set_xlabel("经度 (°E)"); ax.set_ylabel("纬度 (°N)")
    ax.tick_params(labelsize=8)

    # (b) coverage
    ax = ax1
    radii = np.arange(1, 21)
    cov = [fault_match(mask, tx, ty, lon, lat, radius_km=float(r))["recall"] * 100 for r in radii]
    ax.plot(radii, cov, "o-", color=NAVY, lw=1.5, ms=3)
    ax.axvline(5, color=GREY, ls=":", lw=1)
    ax.annotate(f"{cov[4]:.1f}%", (5, cov[4]), xytext=(7, cov[4] - 8),
                fontsize=9, color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
    ax.set_xlabel("缓冲半径 (km)"); ax.set_ylabel("覆盖率 (%)")
    ax.set_title("(b) 断层覆盖率", fontsize=10)
    ax.grid(alpha=0.3); ax.tick_params(labelsize=8)

    # (c) rose
    seg = lineament_segments(mask, lon, lat, 12, 1.5)
    th = np.array([s[0] for s in seg]) if seg else np.array([])
    lw = np.array([s[1] for s in seg]) if seg else np.array([])
    fth = []
    for arr in test:
        for (x0, y0), (x1, y1) in zip(arr[:-1], arr[1:]):
            fth.append(np.degrees(np.arctan2((x1 - x0) * 111.32 * np.cos(np.radians(y0)),
                                             (y1 - y0) * 111.32)) % 180)
    bins = np.arange(0, 181, 15); cen = (bins[:-1] + bins[1:]) / 2
    cnt, _ = np.histogram(th, bins=bins, weights=lw)
    ax2.bar(np.radians(cen), cnt, width=np.radians(13), color=NAVY, alpha=0.8,
            label="线性体")
    if fth:
        cnt2, _ = np.histogram(fth, bins=bins, weights=np.ones(len(fth)))
        ax2.bar(np.radians(cen), cnt2, width=np.radians(13), color=RED, alpha=0.35,
                label="断层")
    ax2.set_theta_zero_location("N"); ax2.set_theta_direction(-1)
    ax2.set_title("(c) 走向玫瑰图", fontsize=10, pad=18)
    ax2.legend(fontsize=6, loc="lower left", bbox_to_anchor=(-0.15, -0.18), frameon=False)
    ax2.tick_params(labelsize=6)

    # (d) cumulative distance
    ax = ax3
    lp = _mask_points_km(mask, lon, lat)
    fp = np.column_stack(_xy(tx, ty))
    tree = cKDTree(fp)
    dist, _ = tree.query(lp)
    sd = np.sort(dist)
    ax.plot(sd, np.arange(1, len(sd) + 1) / len(sd) * 100, color=NAVY, lw=1.6)
    ax.axvline(5, color=GREY, ls=":", lw=1)
    frac5 = float(np.mean(dist <= 5) * 100)
    ax.annotate(f"{frac5:.0f}% ≤5 km", (5, frac5), xytext=(9, frac5 - 18),
                fontsize=8, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
    ax.set_xlim(0, 30); ax.set_ylim(0, 100)
    ax.set_xlabel("线性体到断层距离 (km)"); ax.set_ylabel("累积比例 (%)")
    ax.set_title("(d) 距离累积分布", fontsize=10)
    ax.grid(alpha=0.3); ax.tick_params(labelsize=8)

    fig.tight_layout()
    p = OUT / "fig_paper_fault_validation.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


def inversion_figure():
    f = np.load(ROOT / "output_real" / "test_field_15s_jodc.npz")
    g = f["g15"]; bathy = f["bathy"]; lon = f["lon"]; lat = f["lat"]
    dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = float(lat[1] - lat[0]) * 111320
    z0 = float(-np.median(bathy))
    dg = bp(g, dx, dy, 20e3, 200e3)
    out = parker_inversion(dg, dx, dy, 2200.0, z0,
                           n_iter=8, lam_min=25e3, relax=0.6)
    h = out[0] if isinstance(out, tuple) else out
    target = z0 + bathy
    h_bp = h  # Parker 输出已是 20–200 km 带内
    t_bp = bp(target, dx, dy, 20e3, 200e3)
    deep = bathy < -2000
    inner = np.zeros_like(deep); inner[40:-40, 40:-40] = True
    deep = deep & inner

    fig, axes = plt.subplots(1, 5, figsize=(14.6, 4.15), dpi=220,
                             gridspec_kw={"width_ratios": [1.05, 0.95, 1.15, 0.9, 1.0]})
    # (a) scatter
    ax = axes[0]
    m = deep & np.isfinite(t_bp) & np.isfinite(h_bp)
    x = t_bp[m][::15]; y = h_bp[m][::15]
    hb = ax.hexbin(x, y, gridsize=45, cmap="viridis", bins="log", mincnt=1)
    lo, hi = np.percentile(np.r_[x, y], [1, 99])
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    rmse = float(np.sqrt(np.mean((h_bp[m] - t_bp[m]) ** 2)))
    corr = float(np.corrcoef(h_bp[m], t_bp[m])[0, 1])
    ax.set_xlabel("真值地形 (m)"); ax.set_ylabel("反演地形 (m)")
    ax.set_title("(a) 反演—真值", fontsize=10)
    ax.text(0.05, 0.92, f"RMSE {rmse:.0f} m\ncorr {corr:.2f}", transform=ax.transAxes,
            fontsize=8, va="top", color="white",
            bbox=dict(fc=NAVY, ec="none", alpha=0.8))
    ax.tick_params(labelsize=8)

    # (b) band bars with literature ranges
    ax = axes[1]
    bands = ["20–50", "50–100", "100–200"]
    vals = []
    for lo, hi in ((20e3, 50e3), (50e3, 100e3), (100e3, 200e3)):
        ha = bp(h, dx, dy, lo, hi); ta = bp(target, dx, dy, lo, hi)
        mm = deep & np.isfinite(ha) & np.isfinite(ta)
        vals.append(float(np.sqrt(np.mean((ha[mm] - ta[mm]) ** 2))))
    ax.bar(bands, vals, color=NAVY, width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 8, f"{v:.0f}", ha="center", fontsize=9, color=NAVY)
    ax.fill_between([-0.4, 0.4], 100, 300, color=RED, alpha=0.08)
    ax.fill_between([0.6, 1.4], 200, 400, color=RED, alpha=0.08)
    ax.fill_between([1.6, 2.4], 300, 500, color=RED, alpha=0.08)
    ax.set_ylabel("RMSE (m)"); ax.set_title("(b) 分频精度", fontsize=10)
    ax.set_ylim(0, 420); ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.3)

    # (c) profile along longitude
    ax = axes[2]
    j = int(np.argmin(np.abs(lon - 143.5)))
    ax.plot(lat[40:-40], t_bp[40:-40, j], color="k", lw=1.4, label="真值")
    ax.plot(lat[40:-40], h_bp[40:-40, j], color=RED, lw=1.2, label="反演")
    ax.set_xlabel("纬度 (°N)"); ax.set_ylabel("地形 (m)")
    ax.set_title(f"(c) 剖面 {lon[j]:.1f}°E", fontsize=10)
    ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)

    # (d) residual histogram
    ax = axes[3]
    res = (h_bp - t_bp)[deep & np.isfinite(t_bp) & np.isfinite(h_bp)]
    ax.hist(res, bins=60, color=NAVY, alpha=0.8, density=True)
    mu, sd = float(np.mean(res)), float(np.std(res))
    xx = np.linspace(mu - 4 * sd, mu + 4 * sd, 200)
    ax.plot(xx, np.exp(-0.5 * ((xx - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)),
            color=RED, lw=1.2, label=f"σ={sd:.0f} m")
    ax.set_xlabel("残差 (m)"); ax.set_ylabel("密度")
    ax.set_title("(d) 残差分布", fontsize=10)
    ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)

    # (e) coherence vs wavelength
    ax = axes[4]
    edges = [(20e3, 30e3), (30e3, 50e3), (50e3, 80e3), (80e3, 120e3),
             (120e3, 200e3)]
    cent, coh = [], []
    for lo, hi in edges:
        ha = bp(h, dx, dy, lo, hi); ta = bp(target, dx, dy, lo, hi)
        mm = deep & np.isfinite(ha) & np.isfinite(ta)
        coh.append(float(np.corrcoef(ha[mm], ta[mm])[0, 1]))
        cent.append((lo + hi) / 2e3)
    ax.plot(cent, coh, "o-", color=NAVY, lw=1.5, ms=4)
    ax.axhline(0.5, color=RED, ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_xticks([20, 30, 50, 80, 120, 200])
    ax.set_xticklabels(["20", "30", "50", "80", "120", "200"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylim(0, 1)
    ax.set_xlabel("波长 (km)"); ax.set_ylabel("相干系数")
    ax.set_title("(e) 相干性", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.tick_params(labelsize=8)
    for xv, yv in zip(cent, coh):
        ax.text(xv, min(yv + 0.06, 0.95), f"{yv:.2f}", ha="center",
                fontsize=6.5, color=NAVY)

    fig.tight_layout()
    p = OUT / "fig_paper_inversion_validation.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


def fault_method_figure():
    g, bathy, lon, lat = build_gm9()
    dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = float(lat[1] - lat[0]) * 111320
    k = 30
    b = bandpass(g, dx, dy, 5e3, 20e3)
    b_in = b[k:-k, k:-k]
    t_in = thg(b_in, dx, dy) * 1000.0
    m_in = filter_lineaments(
        lineament_mask(b_in, dx, dy, pct=70) & (bathy[k:-k, k:-k] < 0), 20, 1.8)
    mask = np.zeros_like(b, dtype=bool)
    mask[k:-k, k:-k] = m_in
    lines = []
    for f in json.loads((OUT / "nz_gm9_offshore.json").read_text(encoding="utf-8"))["features"]:
        geo = f.get("geometry") or {}
        if geo.get("type") != "LineString":
            continue
        a = np.asarray(geo["coordinates"], float)
        if ((a[:, 0] >= lon.min()) & (a[:, 0] <= lon.max()) &
                (a[:, 1] >= lat.min()) & (a[:, 1] <= lat.max())).any():
            lines.append(a)

    fig, axes = plt.subplots(1, 4, figsize=(14.6, 3.7), dpi=220)
    ext = [lon[k], lon[-k], lat[k], lat[-k]]
    gi = g[k:-k, k:-k]
    bi = b_in
    ti = t_in
    v = np.percentile(np.abs(gi), 99)
    axes[0].imshow(gi, cmap="RdBu_r", origin="lower", vmin=-v, vmax=v,
                   extent=ext, aspect="auto")
    axes[0].set_title("(a) KF 15″ 场", fontsize=10)
    v = np.percentile(np.abs(bi), 99)
    axes[1].imshow(bi, cmap="RdBu_r", origin="lower", vmin=-v, vmax=v,
                   extent=ext, aspect="auto")
    axes[1].set_title("(b) 5–20 km 带通", fontsize=10)
    vth = np.percentile(ti, 99)
    axes[2].imshow(ti, cmap="inferno", origin="lower", vmin=0, vmax=vth,
                   extent=ext, aspect="auto")
    axes[2].set_title("(c) THG (mGal/km)", fontsize=10)
    axes[2].contour(lon[k:-k], lat[k:-k], m_in, levels=[0.5],
                    colors="cyan", linewidths=0.45)
    depth = np.where(bathy < 0, bathy, 0.0)
    rgb = LightSource(azdeg=315, altdeg=45).shade(
        depth, cmap=plt.cm.Blues_r, vmin=-5000, vmax=0, blend_mode="soft")
    axes[3].imshow(rgb[k:-k, k:-k], origin="lower", extent=ext, aspect="auto")
    for arr in lines:
        axes[3].plot(arr[:, 0], arr[:, 1], color=RED, lw=1.2)
    ys, xs = np.where(mask)
    axes[3].scatter(lon[xs], lat[ys], s=1.6, c="#FFD400", linewidths=0)
    axes[3].set_xlim(ext[0], ext[1]); axes[3].set_ylim(ext[2], ext[3])
    axes[3].set_title("(d) 线性体与断层", fontsize=10)
    for ax in axes:
        ax.set_xlabel("经度 (°E)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("纬度 (°N)", fontsize=8)
    fig.tight_layout()
    p = OUT / "fig_fault_method_4.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


def parker_principle_figure():
    f = np.load(ROOT / "output_real" / "test_field_15s_jodc.npz")
    g = f["g15"]; bathy = f["bathy"]; lon = f["lon"]; lat = f["lat"]
    dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = float(lat[1] - lat[0]) * 111320
    z0 = float(-np.median(bathy))
    dg = bp(g, dx, dy, 20e3, 200e3)
    h, hist = parker_inversion(dg, dx, dy, 2200.0, z0,
                               n_iter=8, lam_min=25e3, relax=0.6, verbose=False)

    fig, axes = plt.subplots(1, 4, figsize=(14.6, 3.7), dpi=220)
    ext = [lon[0], lon[-1], lat[0], lat[-1]]
    v = np.percentile(np.abs(dg), 99)
    axes[0].imshow(dg, cmap="RdBu_r", origin="lower", vmin=-v, vmax=v,
                   extent=ext, aspect="auto")
    axes[0].set_title("(a) 重力 20–200 km", fontsize=10)
    v = np.percentile(h, [2, 98])
    axes[1].imshow(h, cmap="Blues_r", origin="lower", vmin=v[0], vmax=v[1],
                   extent=ext, aspect="auto")
    axes[1].set_title("(b) 反演地形", fontsize=10)
    axes[2].plot(np.arange(1, len(hist) + 1), hist, "o-", color=NAVY, lw=1.5)
    axes[2].set_xlabel("迭代次数"); axes[2].set_ylabel("残差 (mGal)")
    axes[2].set_title("(c) 残差收敛", fontsize=10)
    axes[2].grid(alpha=0.3)
    q = bp(z0 + bathy, dx, dy, 20e3, 200e3)
    inner = np.zeros_like(bathy, dtype=bool)
    inner[40:-40, 40:-40] = True
    m = (bathy < -2000) & inner & np.isfinite(q) & np.isfinite(h)
    axes[3].hexbin(q[m][::15], h[m][::15], gridsize=40, cmap="viridis",
                   bins="log", mincnt=1)
    lo, hi = np.percentile(np.r_[q[m], h[m]], [1, 99])
    axes[3].plot([lo, hi], [lo, hi], "r--", lw=1)
    rmse = float(np.sqrt(np.mean((h[m] - q[m]) ** 2)))
    corr = float(np.corrcoef(h[m], q[m])[0, 1])
    axes[3].set_xlabel("真值地形 (m)"); axes[3].set_ylabel("反演 (m)")
    axes[3].set_title("(d) 反演—真值", fontsize=10)
    axes[3].text(0.05, 0.92, f"RMSE {rmse:.0f} m\ncorr {corr:.2f}",
                 transform=axes[3].transAxes, fontsize=8, va="top", color="white",
                 bbox=dict(fc=NAVY, ec="none", alpha=0.8))
    for ax in axes:
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    p = OUT / "fig_parker_principle.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


def flow_figure():
    fig, ax = plt.subplots(figsize=(12.9, 4.8), dpi=240)
    ax.set_xlim(0, 12.9); ax.set_ylim(0.1, 4.85); ax.axis("off")

    def box(x, y, w, h, txt, fc, fs=8.5, tc="white", ec=NAVY):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.03,rounding_size=0.10",
                                    fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=fs, color=tc, linespacing=1.25)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=11, color=GREY, lw=1.1,
                                     shrinkA=0, shrinkB=0))

    # 数据
    box(0.20, 3.70, 1.75, 0.55, "卫星重力 SSV 1′", "#D6E2F0", tc=NAVY)
    box(0.20, 3.05, 1.75, 0.55, "船测重力\njp_data + JODC", "#D6E2F0", tc=NAVY)
    box(0.20, 2.40, 1.75, 0.55, "地形 GEBCO / GMRT", "#D6E2F0", tc=NAVY)

    # 重建
    box(2.30, 2.95, 1.80, 1.35, "空间 KF\n预测 + 更新\n6 轮 + RTS", NAVY, fs=9)
    box(4.50, 2.95, 1.80, 1.35, "15″ 重力场\n1.85 km 网格", "#2E5C8A", fs=9)

    # 断裂支路
    box(6.85, 1.80, 1.95, 1.00, "带通分解\n5–20 km\nTHG 脊线", "#7A4E2D")
    box(9.10, 1.80, 1.70, 1.00, "线性体\n5–20 km THG", "#8A5A3B")
    box(11.10, 1.80, 1.60, 1.00, "GM9 断层\n5 km 61.6%", "#E8D5C4", tc=NAVY)

    # 反演支路
    box(6.85, 0.45, 1.95, 1.00, "Parker 反演\n20–200 km\n(KF 谱域对照)", "#7A4E2D")
    box(9.10, 0.45, 1.70, 1.00, "海底地形\n20–200 km", "#8A5A3B")
    box(11.10, 0.45, 1.60, 1.00, "独立验证\nRMSE 372 m", "#E8D5C4", tc=NAVY)

    # 左下补充：参数、观测规模、验证数据
    box(0.20, 1.28, 2.85, 1.02,
        "KF 参数\nα=0.5, σ=4 px\nQ=4, R_sat=9, R_ship=1", NAVY, fs=8)
    box(3.20, 1.28, 3.10, 1.02,
        "观测规模\n状态 1.84e7\n卫星观测 1.15e6", NAVY, fs=8)
    box(0.20, 0.22, 6.10, 0.82,
        "验证数据：JODC 船测 · GMRT 地形 · GM9 断层编目",
        "#D6E2F0", fs=8, tc=NAVY)

    # 箭头
    arrow(1.95, 3.95, 2.30, 3.70)
    arrow(1.95, 3.32, 2.30, 3.55)
    arrow(1.95, 2.68, 2.30, 3.15)
    arrow(4.10, 3.62, 4.50, 3.62)
    arrow(6.30, 3.35, 6.85, 2.55)
    arrow(6.30, 2.95, 6.85, 1.20)
    arrow(8.80, 2.30, 9.10, 2.30)
    arrow(10.80, 2.30, 11.10, 2.30)
    arrow(8.80, 0.95, 9.10, 0.95)
    arrow(10.80, 0.95, 11.10, 0.95)

    # 分区标题
    ax.text(0.20, 4.55, "① 数据与重力场重建", fontsize=10.5, color=NAVY,
            fontweight="bold")
    ax.text(6.85, 3.02, "② 断裂识别", fontsize=10.5, color=NAVY,
            fontweight="bold")
    ax.text(6.85, 1.60, "③ 反演", fontsize=10.5, color=NAVY,
            fontweight="bold")
    ax.text(4.55, 4.42, "同一套 15″ 场进入两条支路", fontsize=8.2, color=GREY)
    ax.text(11.10, 3.02, "外部编目", fontsize=8.2, color=GREY)
    ax.text(11.10, 1.60, "独立水深", fontsize=8.2, color=GREY)

    ax.plot([6.55, 6.55], [1.55, 4.40], color="#C9D9EA", lw=1, ls="--")
    ax.plot([10.95, 10.95], [0.30, 4.40], color="#C9D9EA", lw=1, ls="--")

    fig.tight_layout()
    p = OUT / "fig_paper_method_flow.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    fault_figure()
    fault_method_figure()
    inversion_figure()
    parker_principle_figure()
    flow_figure()
