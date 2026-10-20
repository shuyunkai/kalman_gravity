"""三部分演示：

  一、15″ 重力场重建（已有，见 prepare_test_field.py）
  二、多尺度分解 + 断裂属性（向上延拓/THG/tilt/theta + 脊线）
  三、Parker–Oldenburg 反演海底地形（并与 GEBCO 对比）

运行：python src/run_three_parts.py
输出：output_real/three_parts/*.png + report.txt
"""
from pathlib import Path
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.font_manager as fm

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent))

from scale_decomposition import (multi_scale, thg, tilt_angle, theta_map,
                                 ridge_mask, lineament_mask, filter_lineaments,
                                 bandpass as bp_scale)
from bathymetry_inversion import (parker_forward, parker_inversion,
                                  bandpass as bp_inv)
from fault_validation import (lineament_segments, rose_hist, distance_stats,
                              load_usgs_csv, trench_distance, load_gem_faults,
                              densify, fault_match)
from inversion_eval import band_metrics, amplitude_slope, profile
from kf_inversion import spectral_kf_inversion

FONT = Path(r"D:\gravity\fonts\SourceHanSansSC-Regular.otf")
if FONT.exists():
    fm.fontManager.addfont(str(FONT))
plt.rcParams.update({
    "font.sans-serif": ["Source Han Sans SC", "Microsoft YaHei"],
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
})

ROOT = Path(__file__).parent.parent
NPZ = ROOT / "output_real" / "test_field_15s.npz"
OUT = ROOT / "output_real" / "three_parts"
OUT.mkdir(parents=True, exist_ok=True)

RHO = 1640.0          # 海水-地壳密度差 kg/m^3
LAM_MIN = 20e3        # 反演保留的最短波长 20 km
LAM_MAX = 200e3       # 去掉 >200 km 的深部场


def style(ax, title):
    ax.set_title(title, fontsize=11, color="#1F3864", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#C9D9EA")


def main():
    t0 = time.time()
    d = np.load(NPZ)
    g = d["g15"].astype(np.float64)
    bathy = d["bathy"].astype(np.float64)
    lon, lat = d["lon"], d["lat"]
    ny, nx = g.shape

    lat_c = float(np.mean(lat))
    dx = float(lon[1] - lon[0]) * 111320.0 * np.cos(np.radians(lat_c))
    dy = float(lat[1] - lat[0]) * 111320.0
    print(f"grid {ny}x{nx}, dx={dx/1e3:.1f} km, dy={dy/1e3:.1f} km, "
          f"lon {lon[0]:.2f}-{lon[-1]:.2f}, lat {lat[0]:.2f}-{lat[-1]:.2f}")

    g = np.nan_to_num(g, nan=float(np.nanmean(g)))
    bathy = np.nan_to_num(bathy, nan=-4000.0)

    # ---------------------------------------------------------------- 二、尺度分解
    print("\n[2/3] 多尺度分解 + 断裂属性")
    band_defs = [(10e3, 50e3, "10–50 km"), (50e3, 100e3, "50–100 km"),
                 (100e3, 200e3, "100–200 km")]
    bands = {lab: bp_scale(g, dx, dy, lo, hi, order=4) for lo, hi, lab in band_defs}

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.6), dpi=180)
    for ax, (lo, hi, lab) in zip(axes[0], band_defs):
        v = np.percentile(np.abs(bands[lab]), 99)
        im = ax.imshow(bands[lab], cmap="RdBu_r", origin="lower", vmin=-v, vmax=v,
                       interpolation="nearest", aspect="auto")
        style(ax, f"尺度带 {lab}")
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02).set_label("mGal", fontsize=8)
    print("   computing tilt per band")
    for ax, (lo, hi, lab) in zip(axes[1], band_defs):
        t = tilt_angle(bands[lab], dx, dy)
        im = ax.imshow(t, cmap="RdBu_r", origin="lower", vmin=-1.4, vmax=1.4,
                       interpolation="nearest", aspect="auto")
        style(ax, f"tilt angle（{lab}）")
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02).set_label("rad", fontsize=8)
    fig.suptitle("多尺度分解与 tilt 属性（带通 10–200 km，15″ 场）",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_scale_tilt.png", bbox_inches="tight")
    plt.close(fig)

    # 三个尺度带的断裂候选（THG 脊线）
    density = np.zeros_like(g)
    ridge_info = {}
    for lo, hi, lab in band_defs:
        mask = filter_lineaments(lineament_mask(bands[lab], dx, dy, pct=84)
                                 & (bathy < 0), min_area=60, min_elong=2.2)
        density += mask.astype(np.float64)
        ridge_info[lab] = dict(pixels=int(mask.sum()),
                               ratio=float(mask.mean() * 100))
        print(f"   {lab}: lineament pixels={mask.sum():,} ({mask.mean()*100:.2f}%)")

    from scipy.ndimage import gaussian_filter
    dens = gaussian_filter(density, sigma=4)

    fig, ax = plt.subplots(figsize=(9.5, 7.2), dpi=190)
    v = np.percentile(np.abs(g), 99)
    ax.imshow(g, cmap="RdBu_r", origin="lower", vmin=-v, vmax=v, alpha=0.85,
              interpolation="nearest", aspect="auto",
              extent=[lon[0], lon[-1], lat[0], lat[-1]])
    ys, xs = np.where(density > 0)
    m = (xs > 30) & (xs < nx - 30) & (ys > 30) & (ys < ny - 30)
    ax.scatter(lon[xs[m]], lat[ys[m]], s=1.2, c="black", alpha=0.75, linewidths=0)
    ax.set_xlabel("经度 (°E)"); ax.set_ylabel("纬度 (°N)")
    ax.set_title("多尺度断裂候选（三个尺度带，去小块/闭环后）",
                 fontsize=12, fontweight="bold", color="#1F3864")
    fig.tight_layout()
    fig.savefig(OUT / "fig_fault_candidates.png", bbox_inches="tight")
    plt.close(fig)

    # ---- 定量验证：走向、地震、海沟轴
    print("   validation: rose / earthquakes / trench axis")
    ocean = bathy < 0
    masks = {lab: filter_lineaments(lineament_mask(bands[lab], dx, dy, pct=84)
                                    & ocean, min_area=60, min_elong=2.2)
             for _, _, lab in band_defs}
    seg_by_band = {lab: lineament_segments(masks[lab], lon, lat, 80, 2.2)
                   for lab in masks}

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), dpi=180,
                             subplot_kw={"projection": "polar"})
    for ax, (_, _, lab) in zip(axes, band_defs):
        th, hh = rose_hist(seg_by_band[lab])
        ax.bar(th, hh, width=np.radians(15) * 0.88, color="#1F4E79", alpha=0.85)
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        ax.set_title(f"{lab}\n{len(seg_by_band[lab])} 段", fontsize=9)
    fig.suptitle("断裂候选走向玫瑰图（按长度加权）", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_fault_rose.png", bbox_inches="tight")
    plt.close(fig)

    gem_path = OUT / "gem_active_faults.geojson"
    fault_stats = {}
    flon = flat = np.array([])
    if gem_path.exists():
        flines = load_gem_faults(gem_path, lon.min(), lon.max(), lat.min(), lat.max())
        flon, flat = densify(flines, step_km=2.0)
        for lab in masks:
            fault_stats[lab] = fault_match(masks[lab], flon, flat, lon, lat)
        print(f"   GEM faults: {len(flines)} lines, {len(flon):,} points")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6.0), dpi=180,
                                 gridspec_kw={"width_ratios": [1.5, 1]})
        ax = axes[0]
        v = np.percentile(np.abs(g), 99)
        ax.imshow(g, cmap="Greys", origin="lower", vmin=-v, vmax=v, alpha=0.6,
                  interpolation="nearest", aspect="auto",
                  extent=[lon[0], lon[-1], lat[0], lat[-1]])
        for arr in flines:
            ax.plot(arr[:, 0], arr[:, 1], color="#C0392B", lw=1.1, alpha=0.9)
        ys, xs = np.where(masks["10–50 km"])
        ax.scatter(lon[xs], lat[ys], s=0.3, c="#1F4E79", alpha=0.55, linewidths=0)
        ax.set_xlabel("经度 (°E)"); ax.set_ylabel("纬度 (°N)")
        ax.set_title("GEM 活动断层（红）与线性体（蓝）", fontsize=11,
                     fontweight="bold")

        ax = axes[1]
        labs = list(fault_stats)
        rec = [fault_stats[k]["recall"] * 100 for k in labs]
        pre = [fault_stats[k]["precision"] * 100 for k in labs]
        x = np.arange(len(labs)); w = 0.36
        ax.bar(x - w / 2, rec, w, label="断层 20 km 内有线性体", color="#1F4E79")
        ax.bar(x + w / 2, pre, w, label="线性体 20 km 内有断层", color="#2E75B6")
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8.5)
        ax.set_ylabel("%"); ax.set_ylim(0, 100)
        ax.legend(fontsize=8, frameon=False)
        ax.set_title("吻合率（随机基线 %.1f%% / %.1f%%）"
                     % (fault_stats[list(fault_stats)[0]]["recall_random"] * 100,
                        fault_stats[list(fault_stats)[0]]["precision_random"] * 100),
                     fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "fig_fault_match.png", bbox_inches="tight")
        plt.close(fig)

    eq_csv = OUT / "usgs_eq.csv"
    if not eq_csv.exists():
        import urllib.request
        url = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv"
               "&starttime=2000-01-01&endtime=2026-01-01"
               "&minlatitude=32&maxlatitude=40&minlongitude=141&maxlongitude=149"
               "&minmagnitude=4.0&maxdepth=100")
        urllib.request.urlretrieve(url, eq_csv)
    eq_lon, eq_lat, eq_mag = load_usgs_csv(eq_csv)
    print(f"   earthquakes: {len(eq_lon):,}")

    eq_stats = {}
    for lab in masks:
        eq_stats[lab] = distance_stats(masks[lab], eq_lon, eq_lat, lon, lat)
    tr_stats = trench_distance(masks["10–50 km"], bathy, lat, lon)

    lab0 = "10–50 km"
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), dpi=180,
                             gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    v = np.percentile(np.abs(g), 99)
    ax.imshow(g, cmap="Greys", origin="lower", vmin=-v, vmax=v, alpha=0.65,
              interpolation="nearest", aspect="auto",
              extent=[lon[0], lon[-1], lat[0], lat[-1]])
    ys, xs = np.where(masks[lab0])
    ax.scatter(lon[xs], lat[ys], s=0.3, c="#1F4E79", alpha=0.6, linewidths=0)
    ax.scatter(eq_lon, eq_lat, s=np.clip(eq_mag, 1, 7) ** 1.8, c="#C0392B",
               alpha=0.55, edgecolors="none", label="USGS M≥4")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_xlabel("经度 (°E)"); ax.set_ylabel("纬度 (°N)")
    ax.set_title(f"线性体（{lab0}）与地震分布", fontsize=11, fontweight="bold")

    ax = axes[1]
    d = eq_stats[lab0]["dist"]
    ax.hist(d, bins=60, color="#1F4E79", alpha=0.8)
    ax.axvline(eq_stats[lab0]["median_km"], color="#C0392B", lw=1.5,
               label=f"中位数 {eq_stats[lab0]['median_km']:.1f} km")
    ax.axvline(20, color="#7F8C8D", ls="--", lw=1.2, label="20 km")
    ax.set_xlabel("地震点到最近线性体的距离 (km)")
    ax.set_ylabel("地震数")
    ax.set_title(f"距离分布：{eq_stats[lab0]['frac_in_radius']*100:.1f}% 在 20 km 内"
                 f"（随机点 {eq_stats[lab0]['random_frac']*100:.1f}%）",
                 fontsize=9.5)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_fault_validation.png", bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- 三、反演
    print("\n[3/3] Parker–Oldenburg 反演")
    z0 = float(np.mean(-bathy))
    h_true = z0 + bathy
    dg_bp = bp_inv(g, dx, dy, LAM_MIN, LAM_MAX)
    h_bp = bp_inv(h_true, dx, dy, LAM_MIN, LAM_MAX)
    fwd = parker_forward(h_bp, dx, dy, RHO, z0, n_terms=6)
    corr_fwd = float(np.corrcoef(fwd.ravel(), dg_bp.ravel())[0, 1])
    amp_ratio = float(np.std(fwd) / np.std(dg_bp))
    print(f"   reference depth z0={z0:.0f} m")
    print(f"   forward check: corr={corr_fwd:.3f}, amplitude ratio={amp_ratio:.3f}")

    # 密度差扫描（与 pyParkO 的 delta_sigma 探索等价）
    print("   density-contrast sweep:")
    best_rho, best_score = RHO, -9
    for rho in (1640.0, 1800.0, 2000.0, 2200.0):
        h_try, _ = parker_inversion(dg_bp, dx, dy, rho, z0, n_iter=6,
                                    lam_min=25e3, relax=0.6, verbose=False)
        h_try = bp_inv(h_try, dx, dy, LAM_MIN, LAM_MAX)
        r = float(np.corrcoef(h_try.ravel(), h_bp.ravel())[0, 1])
        print(f"     rho={rho:.0f}: corr={r:.3f}")
        if r > best_score:
            best_score, best_rho = r, rho
    print(f"   selected rho={best_rho:.0f} kg/m^3")

    h_inv, hist = parker_inversion(dg_bp, dx, dy, best_rho, z0,
                                   n_iter=8, lam_min=25e3, relax=0.6)
    h_inv_bp = bp_inv(h_inv, dx, dy, LAM_MIN, LAM_MAX)
    rmse = float(np.sqrt(np.mean((h_inv_bp - h_bp) ** 2)))
    corr_h = float(np.corrcoef(h_inv_bp.ravel(), h_bp.ravel())[0, 1])
    print(f"   Parker: RMSE={rmse:.1f} m, corr={corr_h:.3f}")

    # 谱域 KF 反演（主方法）
    h_kf, hist_kf = spectral_kf_inversion(dg_bp, dx, dy, best_rho, z0,
                                          prior_rms=2500.0, noise_mgal=2.0,
                                          n_iter=8, lam_min=25e3, relax=0.6)
    h_kf_bp = bp_inv(h_kf, dx, dy, LAM_MIN, LAM_MAX)
    rmse_kf = float(np.sqrt(np.mean((h_kf_bp - h_bp) ** 2)))
    corr_kf = float(np.corrcoef(h_kf_bp.ravel(), h_bp.ravel())[0, 1])
    print(f"   KF: RMSE={rmse_kf:.1f} m, corr={corr_kf:.3f}")
    h_inv, h_inv_bp = h_kf, h_kf_bp
    rmse, corr_h = rmse_kf, corr_kf
    hist = hist_kf

    bm = band_metrics(h_inv_bp, h_bp, dx, dy)
    amp_a, amp_b = amplitude_slope(h_inv_bp, h_bp)
    h_cal = amp_a * h_inv_bp
    rmse_cal = float(np.sqrt(np.mean((h_cal - h_bp) ** 2)))
    corr_cal = float(np.corrcoef(h_cal.ravel(), h_bp.ravel())[0, 1])
    print(f"   amplitude slope = {amp_a:.2f} -> calibrated RMSE={rmse_cal:.1f} m, "
          f"corr={corr_cal:.3f}")
    dist_pr, hi_pr, ht_pr = profile(h_cal, h_bp, lon, lat,
                                    (143.5, 39.0), (143.5, 33.0))
    print(f"   amplitude slope = {amp_a:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=180,
                             gridspec_kw={"width_ratios": [1.7, 1]})
    ax = axes[0]
    ax.plot(dist_pr, ht_pr, color="#1F3864", lw=1.4, label="GEBCO 带通")
    ax.plot(dist_pr, hi_pr, color="#C0392B", lw=1.2, alpha=0.85, label="反演")
    ax.set_xlabel("沿剖面距离 (km)"); ax.set_ylabel("相对起伏 (m)")
    ax.set_title("横穿海沟剖面（143.5°E，39°→33°N）", fontsize=10.5)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.3)

    ax = axes[1]
    labels = [b["band"] for b in bm]
    vals = [b["rmse"] for b in bm]
    ax.barh(labels, vals, color="#2E75B6")
    for i, b in enumerate(bm):
        ax.text(b["rmse"] + 5, i, f"{b['rmse']:.0f} m / r={b['corr']:.2f}",
                va="center", fontsize=8.5)
    ax.set_xlabel("RMSE (m)")
    ax.set_title("分频段精度", fontsize=10.5)
    ax.set_xlim(0, max(vals) * 1.45)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_inversion_eval.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=180)
    v = np.percentile(np.abs(dg_bp), 99)
    im = axes[0, 0].imshow(dg_bp, cmap="RdBu_r", origin="lower", vmin=-v, vmax=v,
                           aspect="auto")
    style(axes[0, 0], "输入：带通重力异常 Δg（20–200 km）")
    plt.colorbar(im, ax=axes[0, 0], shrink=0.8).set_label("mGal", fontsize=8)

    vh = np.percentile(np.abs(h_bp), 99)
    im = axes[0, 1].imshow(h_bp, cmap="RdBu_r", origin="lower", vmin=-vh, vmax=vh,
                           aspect="auto")
    style(axes[0, 1], "真值：GEBCO 地形带通（相对 z0）")
    plt.colorbar(im, ax=axes[0, 1], shrink=0.8).set_label("m", fontsize=8)

    im = axes[1, 0].imshow(h_inv_bp, cmap="RdBu_r", origin="lower", vmin=-vh, vmax=vh,
                           aspect="auto")
    style(axes[1, 0], f"反演地形（RMSE={rmse:.0f} m, r={corr_h:.2f}）")
    plt.colorbar(im, ax=axes[1, 0], shrink=0.8).set_label("m", fontsize=8)

    ax = axes[1, 1]
    ax.scatter(h_bp.ravel()[::37], h_inv_bp.ravel()[::37], s=1, alpha=0.25,
               color="#1F3864")
    lim = np.percentile(np.abs(h_bp), 99.5)
    ax.plot([-lim, lim], [-lim, lim], color="#C0392B", lw=1)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("GEBCO 带通地形 (m)"); ax.set_ylabel("反演地形 (m)")
    ax.set_title("反演 vs 真值", fontsize=11, color="#1F3864", fontweight="bold")
    ax.grid(alpha=0.3)
    fig.suptitle("谱域 KF 反演海底地形（平稳先验，与 Parker 等价）",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_bathymetry_inversion.png", bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- 报告
    lines = [
        "三部分演示报告",
        "=" * 60,
        f"测试区：lon {lon[0]:.2f}–{lon[-1]:.2f}°E, lat {lat[0]:.2f}–{lat[-1]:.2f}°N",
        f"网格：{ny}×{nx}，15″，dx≈{dx/1e3:.2f} km, dy≈{dy/1e3:.2f} km",
        "",
        "一、15″ 重力场重建（既有）",
        "  BathyPriorKF：SSV 块填充 + α(0.5)×地形布格高频，6 轮 KF + RTS",
        "",
        "二、多尺度分解与断裂属性",
        "  方法：带通 10–50/50–100/100–200 km 三个尺度带（频域），",
        "        THG 阈值(84%，仅海洋) + Zhang–Suen 细化 + 细长体筛选（去小块/闭环）；tilt 作为属性图",
    ]
    for lab, info in ridge_info.items():
        lines.append(f"  {lab}: 线性体像元 {info['pixels']:,}（{info['ratio']:.2f}%）")
    lines += [
        "",
        "  定量验证：",
    ]
    for lab, st in eq_stats.items():
        lines.append(f"    {lab}: 地震点中位距 {st['median_km']:.1f} km，"
                     f"{st['frac_in_radius']*100:.1f}% 在 20 km 内"
                     f"（随机 {st['random_frac']*100:.1f}%），n={st['n']}")
    lines.append(f"    海沟轴到最近线性体：中位 {tr_stats['median_km']:.1f} km，"
                 f"P90 {tr_stats['p90_km']:.1f} km")
    if fault_stats:
        lines.append("  GEM 活动断层吻合率（20 km）：")
        for lab, st in fault_stats.items():
            lines.append(f"    {lab}: 断层被覆盖 {st['recall']*100:.1f}%"
                         f"（随机 {st['recall_random']*100:.1f}%），"
                         f"线性体命中 {st['precision']*100:.1f}%"
                         f"（随机 {st['precision_random']*100:.1f}%）")
    lines += [
        "",
        "三、谱域 KF 反演（Parker 对照）",
        f"  参考深度 z0 = {z0:.0f} m，密度差 ρ = {best_rho:.0f} kg/m³（扫描选定）",
        f"  正演校验：corr = {corr_fwd:.3f}，振幅比 = {amp_ratio:.3f}",
        f"  反演精度：RMSE = {rmse:.1f} m，corr = {corr_h:.3f}，振幅斜率 = {amp_a:.2f}",
        f"  振幅标定后：RMSE = {rmse_cal:.1f} m，corr = {corr_cal:.3f}",
        *[f"    {b['band']}: RMSE {b['rmse']:.0f} m, corr {b['corr']:.2f}, "
          f"振幅比 {b['amp']:.2f}" for b in bm],
        f"  残差收敛：{', '.join(f'{v:.2f}' for v in hist)} mGal",
        "",
        f"图件：{OUT}",
        f"用时：{time.time()-t0:.0f}s",
    ]
    (OUT / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
