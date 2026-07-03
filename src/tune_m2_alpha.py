"""M2 地形先验 KF 大区域验证 + α 参数扫描.

区域: lon=140-150°E, lat=30-40°N (10°×10°, 与 val_visualization 一致)
- 1' 网格: 600×600
- 15" 网格: 2400×2400 = 5.76M 像素
- 验证: MR22-04 独立巡航 (n=16,017)

α 参数扫描: [0.0, 0.5, 1.0, 1.5, 2.0]
  α=0   → 退化为纯 SSV 先验 (相当于原 KF 的双线性先验)
  α=0.5 → 地形高频贡献减半
  α=1.0 → 标准布格重力贡献 (4°×4° 试验最优)
  α=1.5 → 地形高频贡献放大
  α=2.0 → 地形主导

输出:
  - output_real/m2_alpha_sweep.png       — α-RMSE 曲线 + 最优场对比
  - output_real/m2_alpha_sweep_metrics.txt — 数值表
"""
import sys, time
from pathlib import Path
import numpy as np
from scipy.ndimage import zoom as sp_zoom, gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).parent))

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from independent_validation_v2 import (
    load_all_cruises_split, assign_points_to_grid, VAL_CRUISE)
from run_pipeline import load_ssv_region, load_gebco_region
from kf_novel import BathyPriorKF

OUTPUT_DIR = Path(__file__).parent.parent / "output_real"


def get_predictions_at_val(gravity_15s, val_points, lon_hr, lat_hr):
    interp = RegularGridInterpolator(
        (lat_hr, lon_hr), gravity_15s, bounds_error=False, fill_value=np.nan)
    return interp(np.column_stack([val_points[:, 1], val_points[:, 0]]))


def load_bathy_15s(lon_range, lat_range, lon_hr, lat_hr):
    bathy_lr, lon_b, lat_b = load_gebco_region(lon_range, lat_range)
    interp = RegularGridInterpolator(
        (lat_b, lon_b), bathy_lr, bounds_error=False, fill_value=0)
    lon_mesh, lat_mesh = np.meshgrid(lon_hr, lat_hr)
    return interp(np.column_stack([lat_mesh.ravel(), lon_mesh.ravel()])).reshape(
        len(lat_hr), len(lon_hr))


def compute_metrics(pred, truth):
    valid = ~(np.isnan(pred) | np.isnan(truth))
    diff = pred[valid] - truth[valid]
    return {
        'RMSE': float(np.sqrt(np.mean(diff ** 2))),
        'MAE': float(np.mean(np.abs(diff))),
        'Max|e|': float(np.max(np.abs(diff))),
        'STD': float(np.std(diff)),
        'Corr': float(np.corrcoef(pred[valid], truth[valid])[0, 1]),
        'Bias': float(np.mean(diff)),
        'n': int(valid.sum()),
    }


def main(lon_range=(140, 150), lat_range=(30, 40), upscale=4):
    print("=" * 70)
    print("  M2 地形先验 KF — 大区域验证 + α 参数扫描")
    print("=" * 70)
    t_start = time.time()

    alpha_list = [0.0, 0.5, 1.0, 1.5, 2.0]

    # === 数据加载 ===
    print(f"\n[1/4] 加载数据 (区域 lon={lon_range}, lat={lat_range})...")
    gravity_lr, _, lon_lr, lat_lr = load_ssv_region(lon_range, lat_range, load_vgg=False)
    ny_lr, nx_lr = gravity_lr.shape
    ny_hr, nx_hr = ny_lr * upscale, nx_lr * upscale
    lon_hr = np.linspace(lon_range[0], lon_range[1], nx_hr)
    lat_hr = np.linspace(lat_range[0], lat_range[1], ny_hr)
    print(f"  SSV 1': {gravity_lr.shape}  目标 15'': ({ny_hr}, {nx_hr})")

    train_points, val_points = load_all_cruises_split(lon_range, lat_range, VAL_CRUISE)
    print(f"  训练: {len(train_points):,} 点  验证 MR22-04: {len(val_points):,} 点")

    ship_grav = assign_points_to_grid(train_points, lon_hr, lat_hr)
    train_mask_raw = ~np.isnan(ship_grav)
    ssv_hr = sp_zoom(gravity_lr, (upscale, upscale), order=1)
    diff_train_ssv = ship_grav[train_mask_raw] - ssv_hr[train_mask_raw]
    if abs(np.nanmean(diff_train_ssv)) >= 5 or np.nanstd(diff_train_ssv) >= 15:
        ssv_trend_lr = gaussian_filter(gravity_lr, sigma=10)
        ssv_trend_hr = sp_zoom(ssv_trend_lr, (upscale, upscale), order=1)
        ship_grav[train_mask_raw] += ssv_trend_hr[train_mask_raw]
    train_mask = ~np.isnan(ship_grav)
    print(f"  15'' 网格训练点: {train_mask.sum():,}")

    print("  加载 GEBCO 15'' 水深...")
    bathy_15s = load_bathy_15s(lon_range, lat_range, lon_hr, lat_hr)
    print(f"  GEBCO 水深: range=[{bathy_15s.min():.0f}, {bathy_15s.max():.0f}] m")

    region_size = lat_range[1] - lat_range[0]
    val_truth = val_points[:, 2]

    # === α 扫描 ===
    print(f"\n[2/4] α 扫描: {alpha_list}")
    results = {}
    for alpha in alpha_list:
        print(f"\n  --- α = {alpha} ---")
        t0 = time.time()
        m2 = BathyPriorKF(
            hr_grid_shape=(ny_hr, nx_hr), lr_grid_shape=(ny_lr, nx_lr),
            region_size_deg=region_size, alpha=alpha)
        grav_15s = m2.build(
            gravity_lr=gravity_lr, ship_15s=ship_grav, train_mask=train_mask,
            lon_hr=lon_hr, lat_hr=lat_hr, lon_range=lon_range, lat_range=lat_range,
            upscale=upscale, bathy_15s=bathy_15s)
        elapsed = time.time() - t0
        pred = get_predictions_at_val(grav_15s, val_points, lon_hr, lat_hr)
        m = compute_metrics(pred, val_truth)
        m['alpha'] = alpha
        m['time'] = elapsed
        results[alpha] = {'gravity': grav_15s, 'metrics': m, 'pred': pred}
        print(f"  α={alpha}: RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}  "
              f"Max|e|={m['Max|e|']:.2f}  Corr={m['Corr']:.4f}  "
              f"Bias={m['Bias']:.2f}  ({elapsed:.1f}s)")

    # === 双线性基线 ===
    print("\n[3/4] 双线性基线...")
    bi_pred = get_predictions_at_val(ssv_hr, val_points, lon_hr, lat_hr)
    m_bi = compute_metrics(bi_pred, val_truth)
    m_bi['alpha'] = None
    print(f"  双线性: RMSE={m_bi['RMSE']:.2f}  MAE={m_bi['MAE']:.2f}  "
          f"Max|e|={m_bi['Max|e|']:.2f}")

    # === 找最优 α ===
    best_alpha = min(results.keys(), key=lambda a: results[a]['metrics']['RMSE'])
    best = results[best_alpha]
    print(f"\n  最优 α = {best_alpha}  (RMSE={best['metrics']['RMSE']:.2f})")
    print(f"  相对双线性改善: {(1-best['metrics']['RMSE']/m_bi['RMSE'])*100:.1f}%")

    # === 可视化 ===
    print("\n[4/4] 生成图表...")
    fig = plt.figure(figsize=(16, 9))

    # 左: α-RMSE 曲线
    ax1 = fig.add_subplot(2, 3, (1, 2))
    alphas = sorted(results.keys())
    rmses = [results[a]['metrics']['RMSE'] for a in alphas]
    maes = [results[a]['metrics']['MAE'] for a in alphas]
    maxes = [results[a]['metrics']['Max|e|'] for a in alphas]
    ax1.plot(alphas, rmses, 'o-', lw=2, color='#3690C0', label='RMSE', markersize=8)
    ax1.plot(alphas, maes, 's-', lw=2, color='#A6BDDB', label='MAE', markersize=7)
    ax1.plot(alphas, maxes, '^--', lw=1.5, color='#74C476', label='Max|e|', markersize=7)
    ax1.axhline(m_bi['RMSE'], color='red', linestyle=':', lw=1.5,
                label=f'双线性 RMSE={m_bi["RMSE"]:.2f}')
    ax1.axvline(best_alpha, color='gray', linestyle=':', alpha=0.6)
    ax1.scatter([best_alpha], [best['metrics']['RMSE']], s=200, facecolors='none',
                edgecolors='red', linewidths=2, zorder=5,
                label=f'最优 α={best_alpha}')
    ax1.set_xlabel('α (地形高频先验权重)', fontsize=11)
    ax1.set_ylabel('误差 (mGal)', fontsize=11)
    ax1.set_title('α 参数扫描 — M2 地形先验 KF (10°×10° 大区域, MR22-04 验证)',
                  fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(alpha=0.3, linestyle=':')
    ax1.tick_params(labelsize=9)

    # 右上: 指标表
    ax_tbl = fig.add_subplot(2, 3, 3)
    ax_tbl.axis('off')
    cell_text = []
    for a in alphas:
        m = results[a]['metrics']
        impr = (1 - m['RMSE'] / m_bi['RMSE']) * 100
        cell_text.append([f'{a}', f'{m["RMSE"]:.2f}', f'{m["MAE"]:.2f}',
                          f'{m["Max|e|"]:.2f}', f'{m["Corr"]:.4f}',
                          f'{m["Bias"]:.2f}', f'{impr:+.1f}%'])
    cell_text.append(['双线性', f'{m_bi["RMSE"]:.2f}', f'{m_bi["MAE"]:.2f}',
                      f'{m_bi["Max|e|"]:.2f}', f'{m_bi["Corr"]:.4f}',
                      f'{m_bi["Bias"]:.2f}', '(base)'])
    tbl = ax_tbl.table(cellText=cell_text,
                       colLabels=['α', 'RMSE', 'MAE', 'Max|e|', 'Corr', 'Bias', '改善'],
                       loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.4)
    # 高亮最优行
    best_row = alphas.index(best_alpha) + 1
    for j in range(7):
        tbl[best_row, j].set_facecolor('#FFEDA0')
    ax_tbl.set_title('数值对比表 (最优 α 高亮)', fontsize=10, pad=10)

    # 下排: 双线性 / 最优 α / 最差 α 三场对比
    worst_alpha = max(results.keys(), key=lambda a: results[a]['metrics']['RMSE'])

    # 共享色标
    all_gravs = [ssv_hr, results[best_alpha]['gravity'],
                 results[worst_alpha]['gravity']]
    all_vals = np.concatenate([g.ravel() for g in all_gravs])
    all_vals = all_vals[~np.isnan(all_vals)]
    vmax = float(np.percentile(np.abs(all_vals), 99.5))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    extent = [lon_hr[0], lon_hr[-1], lat_hr[0], lat_hr[-1]]

    # 双线性场
    ax2 = fig.add_subplot(2, 3, 4)
    im = ax2.imshow(ssv_hr, cmap='RdBu_r', norm=norm, aspect='equal',
                     origin='lower', extent=extent, interpolation='nearest')
    ax2.plot(val_points[:, 0], val_points[:, 1], 'k-', lw=0.3, alpha=0.5)
    ax2.set_title(f'双线性 (基线)\nRMSE={m_bi["RMSE"]:.2f}', fontsize=10)
    ax2.set_xlabel('经度 (°E)', fontsize=9)
    ax2.set_ylabel('纬度 (°N)', fontsize=9)
    ax2.tick_params(labelsize=8)

    # 最优 α 场
    ax3 = fig.add_subplot(2, 3, 5)
    ax3.imshow(results[best_alpha]['gravity'], cmap='RdBu_r', norm=norm,
                aspect='equal', origin='lower', extent=extent, interpolation='nearest')
    ax3.plot(val_points[:, 0], val_points[:, 1], 'k-', lw=0.3, alpha=0.5)
    bm = results[best_alpha]['metrics']
    ax3.set_title(f'M2 地形先验 KF (α={best_alpha}) ★最优\n'
                  f'RMSE={bm["RMSE"]:.2f} ({(1-bm["RMSE"]/m_bi["RMSE"])*100:+.1f}%)',
                  fontsize=10, fontweight='bold')
    ax3.set_xlabel('经度 (°E)', fontsize=9)
    ax3.tick_params(labelsize=8)

    # 最差 α 场
    ax4 = fig.add_subplot(2, 3, 6)
    ax4.imshow(results[worst_alpha]['gravity'], cmap='RdBu_r', norm=norm,
                aspect='equal', origin='lower', extent=extent, interpolation='nearest')
    ax4.plot(val_points[:, 0], val_points[:, 1], 'k-', lw=0.3, alpha=0.5)
    wm = results[worst_alpha]['metrics']
    ax4.set_title(f'M2 (α={worst_alpha})\nRMSE={wm["RMSE"]:.2f}', fontsize=10)
    ax4.set_xlabel('经度 (°E)', fontsize=9)
    ax4.tick_params(labelsize=8)

    cb = fig.colorbar(im, ax=[ax2, ax3, ax4], orientation='vertical',
                      pad=0.02, shrink=0.85, aspect=30, extend='both')
    cb.set_label('FAA (mGal)', fontsize=10)
    cb.ax.tick_params(labelsize=8)

    plt.tight_layout()
    out_png = OUTPUT_DIR / "m2_alpha_sweep.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  对比图: {out_png}")

    # 保存数值表
    out_txt = OUTPUT_DIR / "m2_alpha_sweep_metrics.txt"
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("  M2 地形先验 KF — α 参数扫描 (10°×10° 大区域)\n")
        f.write(f"  区域: lon={lon_range}, lat={lat_range}\n")
        f.write(f"  验证: MR22-04 (n={len(val_points):,})\n")
        f.write(f"  训练点: {len(train_points):,} (15'' 网格 {train_mask.sum():,} 像素)\n")
        f.write("=" * 80 + "\n")
        f.write(f"  {'α':<6} {'RMSE':<10} {'MAE':<10} {'Max|e|':<10} {'STD':<10} "
                f"{'Corr':<10} {'Bias':<10} {'Impr%':<8} {'Time(s)':<8}\n")
        f.write("-" * 80 + "\n")
        for a in alphas:
            m = results[a]['metrics']
            impr = (1 - m['RMSE'] / m_bi['RMSE']) * 100
            f.write(f"  {a:<6} {m['RMSE']:<10.2f} {m['MAE']:<10.2f} "
                    f"{m['Max|e|']:<10.2f} {m['STD']:<10.2f} {m['Corr']:<10.4f} "
                    f"{m['Bias']:<10.2f} {impr:<+8.1f} {m['time']:<8.1f}\n")
        f.write("-" * 80 + "\n")
        f.write(f"  双线性  {m_bi['RMSE']:<10.2f} {m_bi['MAE']:<10.2f} "
                f"{m_bi['Max|e|']:<10.2f} {m_bi['STD']:<10.2f} {m_bi['Corr']:<10.4f} "
                f"{m_bi['Bias']:<10.2f} {'(base)':<8} {0.0:<8.1f}\n")
        f.write("=" * 80 + "\n")
        f.write(f"  最优 α = {best_alpha}  RMSE={best['metrics']['RMSE']:.2f}  "
                f"改善={(1-best['metrics']['RMSE']/m_bi['RMSE'])*100:.1f}%\n")
    print(f"  指标表: {out_txt}")

    print(f"\n  总用时: {time.time()-t_start:.0f}s")
    return results, best_alpha


if __name__ == "__main__":
    main()
