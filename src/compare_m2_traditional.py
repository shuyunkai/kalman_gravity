"""M2 vs 传统方法 — 大区域精度对比 (3 方法).

区域: lon=132-152°E, lat=26-42°N (20°×16°)
- 1' 网格: 1200×960
- 15" 网格: 4800×3840 = 18.4M 像素
- 验证: MR22-04 独立巡航 (完全未参与训练)

对比方法 (3 个):
  传统基线:
    1. 双线性插值 (SSV 1' 上采样)
    2. 频域融合 (Butterworth 滤波器, SSV 低频 + 船载高频)
  KF 主方法:
    3. M2 地形先验 KF (BathyPriorKF α=0.5)

所有方法在同一 MR22-04 验证集上评估, 保证公平.
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
from freq_fusion_baseline import freq_fusion_reconstruct
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


def load_region_data(lon_range, lat_range, val_cruise=VAL_CRUISE, upscale=4):
    """加载区域数据: SSV 1' + 船载 (按 val_cruise 切分) + GEBCO 15'' 水深."""
    gravity_lr, _, lon_lr, lat_lr = load_ssv_region(lon_range, lat_range, load_vgg=False)
    ny_lr, nx_lr = gravity_lr.shape
    ny_hr, nx_hr = ny_lr * upscale, nx_lr * upscale
    lon_hr = np.linspace(lon_range[0], lon_range[1], nx_hr)
    lat_hr = np.linspace(lat_range[0], lat_range[1], ny_hr)
    print(f"  SSV 1': {gravity_lr.shape}  目标 15'': ({ny_hr}, {nx_hr}) "
          f"({ny_hr*nx_hr/1e6:.1f}M 像素)")

    train_points, val_points = load_all_cruises_split(lon_range, lat_range, val_cruise)
    print(f"  训练: {len(train_points):,} 点  验证 {val_cruise}: {len(val_points):,} 点")

    ship_grav = assign_points_to_grid(train_points, lon_hr, lat_hr)
    train_mask_raw = ~np.isnan(ship_grav)
    ssv_hr = sp_zoom(gravity_lr, (upscale, upscale), order=1)
    # 趋势恢复: 若船载与 SSV 系统偏差大, 用 SSV 平滑趋势补偿
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

    return {
        'gravity_lr': gravity_lr, 'lon_lr': lon_lr, 'lat_lr': lat_lr,
        'lon_hr': lon_hr, 'lat_hr': lat_hr,
        'ny_lr': ny_lr, 'nx_lr': nx_lr, 'ny_hr': ny_hr, 'nx_hr': nx_hr,
        'ship_grav': ship_grav, 'train_mask': train_mask,
        'bathy_15s': bathy_15s, 'ssv_hr': ssv_hr,
        'train_points': train_points, 'val_points': val_points,
    }


def build_three_methods(data, alpha=0.5, n_iter=6, smooth_sigma=15.0):
    """构建 3 方法: 双线性 / 频域融合 / M2."""
    ssv_hr = data['ssv_hr']
    gravity_lr = data['gravity_lr']
    ship_grav = data['ship_grav']
    train_mask = data['train_mask']
    bathy_15s = data['bathy_15s']
    lon_hr, lat_hr = data['lon_hr'], data['lat_hr']
    lon_range, lat_range = data.get('lon_range'), data.get('lat_range')
    ny_hr, nx_hr = data['ny_hr'], data['nx_hr']
    ny_lr, nx_lr = data['ny_lr'], data['nx_lr']
    region_size = lat_range[1] - lat_range[0]

    models = {}
    times = {}

    # 1. 双线性
    t0 = time.time()
    models['双线性插值'] = ssv_hr.copy()
    times['双线性插值'] = time.time() - t0
    print(f"  双线性: {times['双线性插值']:.1f}s")

    # 2. 频域融合
    t0 = time.time()
    print("  频域融合...")
    models['频域融合'] = freq_fusion_reconstruct(
        ssv_hr=ssv_hr, ship_grav=ship_grav, train_mask=train_mask,
        lon_hr=lon_hr, lat_hr=lat_hr,
        lon_range=lon_range, lat_range=lat_range)
    times['频域融合'] = time.time() - t0
    print(f"  频域融合: {times['频域融合']:.1f}s")

    # 3. M2 地形先验 KF
    t0 = time.time()
    print(f"  M2 地形先验 KF (α={alpha})...")
    m2 = BathyPriorKF(
        hr_grid_shape=(ny_hr, nx_hr), lr_grid_shape=(ny_lr, nx_lr),
        region_size_deg=region_size, alpha=alpha)
    models['M2 地形先验KF'] = m2.build(
        gravity_lr=gravity_lr, ship_15s=ship_grav, train_mask=train_mask,
        lon_hr=lon_hr, lat_hr=lat_hr, lon_range=lon_range, lat_range=lat_range,
        upscale=4, bathy_15s=bathy_15s,
        n_iter=n_iter, smooth_sigma=smooth_sigma)
    times['M2 地形先验KF'] = time.time() - t0
    print(f"  M2: {times['M2 地形先验KF']:.1f}s")

    return models, times


def plot_comparison(models, times, metrics_rows, data, save_png, save_txt,
                    lon_range, lat_range, val_cruise_label):
    lon_hr, lat_hr = data['lon_hr'], data['lat_hr']
    val_points = data['val_points']
    train_points = data['train_points']

    method_names = [n for n in models.keys() if models[n] is not None]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.ravel()

    all_vals = np.concatenate([models[n].ravel() for n in method_names])
    all_vals = all_vals[~np.isnan(all_vals)]
    vmax = float(np.percentile(np.abs(all_vals), 99.5))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    extent = [lon_hr[0], lon_hr[-1], lat_hr[0], lat_hr[-1]]

    bi_m = next(m for n, m in metrics_rows if n == '双线性插值')
    kf_methods = [(n, m) for n, m in metrics_rows if n != '双线性插值']
    best_name, best_m = min(kf_methods, key=lambda x: x[1]['RMSE'])

    for ax, (name, m) in zip(axes, metrics_rows):
        is_best = (name == best_name)
        is_bi = (name == '双线性插值')
        im = ax.imshow(models[name], cmap='RdBu_r', norm=norm, aspect='equal',
                       origin='lower', extent=extent, interpolation='nearest')
        ax.plot(val_points[:, 0], val_points[:, 1], 'k-', lw=0.3, alpha=0.5)
        ax.scatter(train_points[:, 0], train_points[:, 1],
                   s=0.15, c='yellow', alpha=0.2, edgecolors='none')
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
        ax.set_title(title, fontsize=11,
                     fontweight='bold' if is_best else 'normal')
        ax.tick_params(labelsize=8)

    # 第 4 个面板: RMSE 柱状图
    ax_bar = axes[3]
    kf_names = [n for n, _ in metrics_rows if n != '双线性插值']
    rmses = [m['RMSE'] for n, m in metrics_rows if n != '双线性插值']
    bi_rmse = bi_m['RMSE']
    y_pos = np.arange(len(kf_names))
    colors = ['#A6BDDB'] * len(kf_names)
    if best_name in kf_names:
        best_idx = kf_names.index(best_name)
        colors[best_idx] = '#3690C0'
    bars = ax_bar.barh(y_pos, rmses, color=colors, edgecolor='none', height=0.55)
    ax_bar.axvline(bi_rmse, color='red', linestyle='--', lw=1.2,
                   label=f'双线性 RMSE={bi_rmse:.2f}')
    ax_bar.invert_yaxis()
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(kf_names, fontsize=9)
    ax_bar.set_xlabel('RMSE (mGal)', fontsize=9)
    ax_bar.set_title('M2 vs 频域融合 RMSE', fontsize=11, fontweight='bold')
    for i, (v, bar) in enumerate(zip(rmses, bars)):
        ax_bar.text(v + 0.05, bar.get_y() + bar.get_height()/2,
                    f'{v:.2f}', va='center', fontsize=10,
                    fontweight='bold' if i == best_idx else 'normal')
    ax_bar.legend(fontsize=8, loc='lower right')
    for spine in ['top', 'right', 'left']:
        ax_bar.spines[spine].set_visible(False)
    ax_bar.grid(axis='x', alpha=0.3)

    cb = fig.colorbar(im, ax=axes[:3].tolist(), orientation='vertical',
                      pad=0.015, shrink=0.85, aspect=30, extend='both')
    cb.set_label('FAA (mGal)', fontsize=10)

    fig.suptitle(f'M2 地形先验 KF vs 传统方法 — 大区域 (20°×16°) 对比\n'
                 f'黑线={val_cruise_label} 验证航迹, 黄点=训练点',
                 fontsize=12, y=1.00)
    plt.savefig(save_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  对比图: {save_png}")

    # 数值表
    with open(save_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write(f"  M2 vs 传统方法 — 大区域精度对比 ({lat_range[1]-lat_range[0]}°×{lon_range[1]-lon_range[0]}°)\n")
        f.write(f"  区域: lon={lon_range}, lat={lat_range}\n")
        f.write(f"  验证: {val_cruise_label} (n={len(val_points):,}, 完全独立未参与训练)\n")
        f.write(f"  训练点: {len(train_points):,}\n")
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
    print(f"  指标表: {save_txt}")


def main(lon_range=(132, 152), lat_range=(26, 42), upscale=4,
         alpha=0.5, val_cruise=VAL_CRUISE):
    print("=" * 70)
    print("  M2 vs 传统方法 — 大区域精度对比 (3 方法)")
    print("=" * 70)
    t_start = time.time()

    print(f"\n[1/3] 加载数据 (区域 lon={lon_range}, lat={lat_range})...")
    data = load_region_data(lon_range, lat_range, val_cruise=val_cruise, upscale=upscale)
    data['lon_range'] = lon_range
    data['lat_range'] = lat_range

    print("\n[2/3] 构建 3 个模型...")
    models, times = build_three_methods(data, alpha=alpha)

    print(f"\n[3/3] 独立验证 ({val_cruise}, n={len(data['val_points']):,})...")
    method_names = list(models.keys())
    val_truth = data['val_points'][:, 2]
    predictions = [get_predictions_at_val(models[n], data['val_points'],
                                          data['lon_hr'], data['lat_hr'])
                   for n in method_names]

    print("\n  " + "=" * 90)
    print(f"  {'Method':<22} {'RMSE':<10} {'MAE':<10} {'Max|e|':<10} "
          f"{'STD':<10} {'Corr':<10} {'Bias':<10} {'Time(s)':<8}")
    print("  " + "-" * 90)
    metrics_rows = []
    for name, pred in zip(method_names, predictions):
        m = compute_metrics(pred, val_truth)
        m['time'] = times[name]
        metrics_rows.append((name, m))
        print(f"  {name:<22} {m['RMSE']:<10.2f} {m['MAE']:<10.2f} {m['Max|e|']:<10.2f} "
              f"{m['STD']:<10.2f} {m['Corr']:<10.4f} {m['Bias']:<10.2f} {times[name]:<8.1f}")

    save_png = OUTPUT_DIR / "m2_vs_traditional.png"
    save_txt = OUTPUT_DIR / "m2_vs_traditional_metrics.txt"
    plot_comparison(models, times, metrics_rows, data, save_png, save_txt,
                    lon_range, lat_range, val_cruise)

    print(f"\n  总用时: {time.time()-t_start:.0f}s")
    return models, metrics_rows


if __name__ == "__main__":
    main()
