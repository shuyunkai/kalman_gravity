"""三种改进 KF 方法 vs 双线性/M2 — 大区域对比试验.

区域: lon=138-152°E, lat=28-42°N (14°×14°, 较前次 10°×10° 扩大约 2×)
- 1' 网格: 840×840
- 15" 网格: 3360×3360 = 11.3M 像素
- 验证: MR22-04 独立巡航

对比方法:
  1. 双线性插值 (基线)
  2. M2 (布格近似, α=0.5, 之前最优)
  3. A: 自适应 α
  4. B: Parker 正演先验
  5. C: 频域三源 KF

输出:
  - output_real/kf_v2_comparison.png       — 5 面板对比图
  - output_real/kf_v2_metrics.txt          — 数值表
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
from kf_novel_v2 import AdaptiveAlphaKF, ParkerPriorKF, MultiSourceKF

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


def main(lon_range=(138, 152), lat_range=(28, 42), upscale=4):
    print("=" * 70)
    print("  三种改进 KF 方法 vs 双线性/M2 — 大区域对比试验")
    print("=" * 70)
    t_start = time.time()

    # === 数据加载 ===
    print(f"\n[1/4] 加载数据 (区域 lon={lon_range}, lat={lat_range})...")
    gravity_lr, _, lon_lr, lat_lr = load_ssv_region(lon_range, lat_range, load_vgg=False)
    ny_lr, nx_lr = gravity_lr.shape
    ny_hr, nx_hr = ny_lr * upscale, nx_lr * upscale
    lon_hr = np.linspace(lon_range[0], lon_range[1], nx_hr)
    lat_hr = np.linspace(lat_range[0], lat_range[1], ny_hr)
    print(f"  SSV 1': {gravity_lr.shape}  目标 15'': ({ny_hr}, {nx_hr}) "
          f"({ny_hr*nx_hr/1e6:.1f}M 像素)")

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

    # === 构建模型 ===
    print("\n[2/4] 构建 5 个模型...")
    models = {}
    times = {}

    # 双线性基线
    t0 = time.time()
    models['双线性'] = ssv_hr.copy()
    times['双线性'] = time.time() - t0
    print(f"  双线性: {times['双线性']:.1f}s")

    # M2 (Bouguer α=0.5, 之前最优)
    t0 = time.time()
    print("\n  --- M2 (布格 α=0.5, baseline 最优) ---")
    m2 = BathyPriorKF(
        hr_grid_shape=(ny_hr, nx_hr), lr_grid_shape=(ny_lr, nx_lr),
        region_size_deg=region_size, alpha=0.5)
    models['M2 布格α=0.5'] = m2.build(
        gravity_lr=gravity_lr, ship_15s=ship_grav, train_mask=train_mask,
        lon_hr=lon_hr, lat_hr=lat_hr, lon_range=lon_range, lat_range=lat_range,
        upscale=upscale, bathy_15s=bathy_15s)
    times['M2 布格α=0.5'] = time.time() - t0
    print(f"  M2 完成: {times['M2 布格α=0.5']:.1f}s")

    # 方法 A: 自适应 α (修正: α_max 降到 0.8, 避免退化区)
    t0 = time.time()
    print("\n  --- 方法 A: 自适应 α (α_max=0.8) ---")
    mA = AdaptiveAlphaKF(
        hr_grid_shape=(ny_hr, nx_hr), lr_grid_shape=(ny_lr, nx_lr),
        region_size_deg=region_size, alpha_max=0.8, alpha_min=0.2,
        block_size_deg=2.0)
    models['A 自适应α'] = mA.build(
        gravity_lr=gravity_lr, ship_15s=ship_grav, train_mask=train_mask,
        lon_hr=lon_hr, lat_hr=lat_hr, lon_range=lon_range, lat_range=lat_range,
        upscale=upscale, bathy_15s=bathy_15s)
    times['A 自适应α'] = time.time() - t0
    print(f"  A 完成: {times['A 自适应α']:.1f}s")

    # 方法 B: Parker 正演先验 (修正: α=1.0, 匹配 Parker 较小 σ)
    t0 = time.time()
    print("\n  --- 方法 B: Parker 正演先验 (α=1.0) ---")
    mB = ParkerPriorKF(
        hr_grid_shape=(ny_hr, nx_hr), lr_grid_shape=(ny_lr, nx_lr),
        region_size_deg=region_size, alpha=1.0)
    models['B Parker先验'] = mB.build(
        gravity_lr=gravity_lr, ship_15s=ship_grav, train_mask=train_mask,
        lon_hr=lon_hr, lat_hr=lat_hr, lon_range=lon_range, lat_range=lat_range,
        upscale=upscale, bathy_15s=bathy_15s)
    times['B Parker先验'] = time.time() - t0
    print(f"  B 完成: {times['B Parker先验']:.1f}s")

    # 方法 C: 三源 KF (修正: SSV 高频预填 + 中频 Parker 替换)
    t0 = time.time()
    print("\n  --- 方法 C: 频域三源 KF (SSV 低 + Parker 中 + SSV 高) ---")
    mC = MultiSourceKF(
        hr_grid_shape=(ny_hr, nx_hr), lr_grid_shape=(ny_lr, nx_lr),
        region_size_deg=region_size, alpha_mid=1.0)
    models['C 三源KF'] = mC.build(
        gravity_lr=gravity_lr, ship_15s=ship_grav, train_mask=train_mask,
        lon_hr=lon_hr, lat_hr=lat_hr, lon_range=lon_range, lat_range=lat_range,
        upscale=upscale, bathy_15s=bathy_15s,
        n_iter=6, smooth_sigma=15.0)
    times['C 三源KF'] = time.time() - t0
    print(f"  C 完成: {times['C 三源KF']:.1f}s")

    # === 评估 ===
    print(f"\n[3/4] 独立验证 (MR22-04, n={len(val_points):,})...")
    method_names = list(models.keys())
    val_truth = val_points[:, 2]
    predictions = [get_predictions_at_val(models[n], val_points, lon_hr, lat_hr)
                   for n in method_names]

    print("\n  " + "=" * 90)
    print(f"  {'Method':<22} {'RMSE':<10} {'MAE':<10} {'Max|e|':<10} "
          f"{'Corr':<10} {'Bias':<10} {'Time(s)':<8}")
    print("  " + "-" * 90)
    metrics_rows = []
    for name, pred in zip(method_names, predictions):
        m = compute_metrics(pred, val_truth)
        m['time'] = times[name]
        metrics_rows.append((name, m))
        print(f"  {name:<22} {m['RMSE']:<10.2f} {m['MAE']:<10.2f} {m['Max|e|']:<10.2f} "
              f"{m['Corr']:<10.4f} {m['Bias']:<10.2f} {times[name]:<8.1f}")

    # === 可视化 ===
    print("\n[4/4] 生成对比图...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    axes = axes.ravel()

    all_vals = np.concatenate([m.ravel() for m in models.values()])
    all_vals = all_vals[~np.isnan(all_vals)]
    vmax = float(np.percentile(np.abs(all_vals), 99.5))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    extent = [lon_hr[0], lon_hr[-1], lat_hr[0], lat_hr[-1]]

    # 找最优 (除双线性外)
    kf_methods = [(n, m) for n, m in metrics_rows if n != '双线性']
    best_name, best_m = min(kf_methods, key=lambda x: x[1]['RMSE'])
    bi_m = next(m for n, m in metrics_rows if n == '双线性')

    for ax, (name, m) in zip(axes, metrics_rows):
        is_best = (name == best_name)
        is_bi = (name == '双线性')
        im = ax.imshow(models[name], cmap='RdBu_r', norm=norm, aspect='equal',
                       origin='lower', extent=extent, interpolation='nearest')
        ax.plot(val_points[:, 0], val_points[:, 1], 'k-', lw=0.3, alpha=0.5)
        ax.scatter(train_points[:, 0], train_points[:, 1],
                   s=0.2, c='yellow', alpha=0.2, edgecolors='none')
        ax.set_xlim(lon_range)
        ax.set_ylim(lat_range)
        ax.set_xlabel('经度 (°E)', fontsize=9)
        ax.set_ylabel('纬度 (°N)', fontsize=9)
        impr = (1 - m['RMSE'] / bi_m['RMSE']) * 100
        title = f'{name}\nRMSE={m["RMSE"]:.2f}'
        if is_best:
            title += f' ★最优 ({impr:+.1f}%)'
        elif is_bi:
            title += ' (基线)'
        else:
            title += f' ({impr:+.1f}%)'
        ax.set_title(title, fontsize=10,
                     fontweight='bold' if is_best else 'normal')
        ax.tick_params(labelsize=8)

    # 第 6 个面板: 指标柱状图
    ax_bar = axes[5]
    ax_bar.axis('off')
    kf_names = [n for n, _ in metrics_rows if n != '双线性']
    rmses = [m['RMSE'] for n, m in metrics_rows if n != '双线性']
    bi_rmse = bi_m['RMSE']
    y_pos = np.arange(len(kf_names))
    colors = ['#A6BDDB'] * len(kf_names)
    best_idx = kf_names.index(best_name)
    colors[best_idx] = '#3690C0'
    bars = ax_bar.barh(y_pos, rmses, color=colors, edgecolor='none', height=0.55)
    ax_bar.axvline(bi_rmse, color='red', linestyle='--', lw=1.2,
                    label=f'双线性 RMSE={bi_rmse:.2f}')
    ax_bar.invert_yaxis()
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(kf_names, fontsize=9)
    ax_bar.set_xlabel('RMSE (mGal)', fontsize=9)
    ax_bar.set_title('KF 方法 RMSE 对比', fontsize=10, fontweight='bold')
    for i, (v, bar) in enumerate(zip(rmses, bars)):
        ax_bar.text(v + 0.05, bar.get_y() + bar.get_height()/2,
                    f'{v:.2f}', va='center', fontsize=9,
                    fontweight='bold' if i == best_idx else 'normal')
    ax_bar.legend(fontsize=8, loc='lower right')
    for spine in ['top', 'right', 'left']:
        ax_bar.spines[spine].set_visible(False)
    ax_bar.grid(axis='x', alpha=0.3)

    cb = fig.colorbar(im, ax=axes[:5].tolist(), orientation='vertical',
                      pad=0.015, shrink=0.85, aspect=35, extend='both')
    cb.set_label('FAA (mGal)', fontsize=10)

    fig.suptitle('三种改进 KF 方法 vs 双线性/M2 — 大区域 (14°×14°) 对比\n'
                 '黑线=MR22-04 验证航迹, 黄点=训练点',
                 fontsize=12, y=1.00)
    out_png = OUTPUT_DIR / "kf_v2_comparison.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  对比图: {out_png}")

    # 保存数值表
    out_txt = OUTPUT_DIR / "kf_v2_metrics.txt"
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write("  三种改进 KF 方法 vs 双线性/M2 — 大区域对比 (14°×14°)\n")
        f.write(f"  区域: lon={lon_range}, lat={lat_range}\n")
        f.write(f"  验证: MR22-04 (n={len(val_points):,})\n")
        f.write(f"  训练点: {len(train_points):,} (15'' 网格 {train_mask.sum():,} 像素)\n")
        f.write("=" * 90 + "\n")
        f.write(f"  {'Method':<22} {'RMSE':<10} {'MAE':<10} {'Max|e|':<10} "
                f"{'STD':<10} {'Corr':<10} {'Bias':<10} {'Impr%':<8} {'Time(s)':<8}\n")
        f.write("-" * 90 + "\n")
        for name, m in metrics_rows:
            impr = (1 - m['RMSE'] / bi_m['RMSE']) * 100
            f.write(f"  {name:<22} {m['RMSE']:<10.2f} {m['MAE']:<10.2f} "
                    f"{m['Max|e|']:<10.2f} {m['STD']:<10.2f} {m['Corr']:<10.4f} "
                    f"{m['Bias']:<10.2f} {impr:<+8.1f} {m['time']:<8.1f}\n")
        f.write("=" * 90 + "\n")
        f.write(f"  最优: {best_name}  RMSE={best_m['RMSE']:.2f}  "
                f"改善={(1-best_m['RMSE']/bi_m['RMSE'])*100:.1f}%\n")
    print(f"  指标表: {out_txt}")

    print(f"\n  总用时: {time.time()-t_start:.0f}s")
    return metrics_rows


if __name__ == "__main__":
    main()
