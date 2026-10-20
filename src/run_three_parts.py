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
        mask = filter_lineaments(lineament_mask(bands[lab], dx, dy, pct=86),
                                 min_area=80, min_elong=2.2)
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
    print(f"   inversion: RMSE={rmse:.1f} m, corr={corr_h:.3f}")

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
    fig.suptitle("Parker–Oldenburg 反演海底地形（ρ=1640 kg/m³，20–200 km 带通）",
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
        "        THG 阈值(86%) + Zhang–Suen 细化 + 细长体筛选（去小块/闭环）；tilt 作为属性图",
    ]
    for lab, info in ridge_info.items():
        lines.append(f"  {lab}: 脊线像元 {info['pixels']:,}（{info['ratio']:.2f}%）")
    lines += [
        "",
        "三、Parker–Oldenburg 反演",
        f"  参考深度 z0 = {z0:.0f} m，密度差 ρ = {best_rho:.0f} kg/m³（扫描选定）",
        f"  正演校验：corr = {corr_fwd:.3f}，振幅比 = {amp_ratio:.3f}",
        f"  反演精度：RMSE = {rmse:.1f} m，corr = {corr_h:.3f}",
        f"  残差收敛：{', '.join(f'{v:.2f}' for v in hist)} mGal",
        "",
        f"图件：{OUT}",
        f"用时：{time.time()-t0:.0f}s",
    ]
    (OUT / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
