"""15" 模型 vs 船载重力真值 — 可视化对比.

对比机制说明:
  - 船载重力 (MR22-04 巡航) = 真值 (船载重力仪精度 ~1-2 mGal)
  - 15" 模型网格 = 待评估对象
  - 对比方式: 在每个船载验证点 (lon, lat, faa_ship) 处,
    从 15" 模型网格双线性插值取模型值 faa_model,
    计算 diff = faa_model - faa_ship
  - RMSE = sqrt(mean(diff²)) = 模型相对船载真值的精度

验证设计 (独立数据验证):
  - 训练集: 30 个日本巡航 (KH-20-10, KR06-12, ..., YK18-08 等),
    原始 10s 采样点精确像素赋值到 15" 网格
  - 验证集: MR22-04 巡航 (R/V Mirai, 2022 年), 完全独立未参与训练
  - 验证区域: 日本海沟及周边 (lon=140°E-150°E, lat=30°N-40°N)
    该区域重力场复杂 (海沟、岛弧、深海平原), 是模型精度的严苛测试场

输出图:
  1. val_scatter_model_vs_ship.png  — 散点图: 模型 vs 船载真值
  2. val_error_histogram.png        — 误差直方图 + 正态拟合
  3. val_track_profile.png          — 沿 MR22-04 航迹剖面
  4. val_region_map.png             — 验证区域地图: 训练/验证巡航航迹
  5. val_spatial_error_map.png      — 误差空间分布 (每个方法)
  6. val_method_diff_map.png        — KF 相对 Bilinear 的空间改进
  7. val_qq_plot.png                — Q-Q 正态性检验
  8. val_error_vs_distance.png      — 误差 vs 距最近训练点距离
  9. val_metrics_bar.png            — 多指标柱状图对比
"""
import sys
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom as sp_zoom, gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from independent_validation_v2 import (
    load_all_cruises_split, assign_points_to_grid, VAL_CRUISE,
    evaluate_15s_raw, JP_DATA_DIR)
from run_pipeline import load_ssv_region
from kriging_baseline import _subsample_train_points
from freq_fusion_baseline import butterworth_2d_filter
from kalman_gravity import KalmanGravityReconstructor
from scipy.spatial import cKDTree

OUTPUT_DIR = Path(__file__).parent.parent / "output_real"


def get_model_predictions_at_val(gravity_15s, val_points, lon_hr, lat_hr):
    """从 15" 模型网格双线性插值到验证点上."""
    interp = RegularGridInterpolator(
        (lat_hr, lon_hr), gravity_15s, bounds_error=False, fill_value=np.nan)
    pred = interp(np.column_stack([val_points[:, 1], val_points[:, 0]]))
    return pred


def plot_scatter(val_truth, predictions, method_names, save_path):
    """散点图: 模型值 vs 船载真值."""
    n = len(method_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    rng = [min(val_truth.min(), min(p.min() for p in predictions)),
           max(val_truth.max(), max(p.max() for p in predictions))]
    for ax, pred, name in zip(axes, predictions, method_names):
        valid = ~(np.isnan(pred) | np.isnan(val_truth))
        p, t = pred[valid], val_truth[valid]
        ax.scatter(t, p, s=2, alpha=0.3, c='steelblue')
        ax.plot(rng, rng, 'r--', lw=1.5, label='y=x (完美)')
        rmse = np.sqrt(np.mean((p - t) ** 2))
        bias = np.mean(p - t)
        corr = np.corrcoef(p, t)[0, 1]
        ax.set_xlabel('船载重力 FAA (mGal) — 真值')
        ax.set_ylabel('模型 FAA (mGal)')
        ax.set_title(f'{name}\nRMSE={rmse:.2f}  Bias={bias:.2f}  R={corr:.4f}')
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xlim(rng)
        ax.set_ylim(rng)
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  散点图: {save_path}")


def plot_error_hist(val_truth, predictions, method_names, save_path):
    """误差直方图: diff = model - ship 分布."""
    n = len(method_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    all_diffs = []
    for pred in predictions:
        valid = ~(np.isnan(pred) | np.isnan(val_truth))
        all_diffs.append(pred[valid] - val_truth[valid])
    x_max = max(np.percentile(np.abs(d), 99) for d in all_diffs)
    bins = np.linspace(-x_max, x_max, 60)
    for ax, diff, name in zip(axes, all_diffs, method_names):
        ax.hist(diff, bins=bins, density=True, alpha=0.7,
                color='steelblue', edgecolor='white')
        ax.axvline(0, color='red', linestyle='--', lw=1.5, label='零线')
        ax.axvline(diff.mean(), color='orange', linestyle='-',
                   lw=1.5, label=f'Bias={diff.mean():.2f}')
        mu, std = diff.mean(), diff.std()
        x = np.linspace(-x_max, x_max, 200)
        y = np.exp(-(x - mu) ** 2 / (2 * std ** 2)) / (std * np.sqrt(2 * np.pi))
        ax.plot(x, y, 'k-', lw=1.5,
                label=f'N({mu:.2f}, {std:.2f})')
        ax.set_xlabel('误差 = 模型 - 船载 (mGal)')
        ax.set_ylabel('概率密度')
        ax.set_title(f'{name}\nSTD={std:.2f} mGal')
        ax.legend(fontsize=8)
        ax.set_xlim(-x_max, x_max)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  误差直方图: {save_path}")


def plot_track_profile(val_points, predictions, method_names, save_path):
    """沿航迹剖面: 验证点序列上模型 vs 船载."""
    # Sort by time (already in chronological order in .grv)
    # Use point index as x-axis
    n_pts = len(val_points)
    # Subsample for clarity (every 50th point)
    step = max(1, n_pts // 400)
    idx = np.arange(0, n_pts, step)
    truth = val_points[idx, 2]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1.5]})
    ax1, ax2 = axes

    ax1.plot(idx, truth, 'k-', lw=1.5, label='船载真值 (MR22-04)', alpha=0.8)
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple', 'tab:orange']
    for pred, name, c in zip(predictions, method_names, colors):
        # pred is full-length (n_pts), subsample to idx
        ax1.plot(idx, pred[idx], '-', lw=1.0, alpha=0.7, label=name, color=c)
        # Error panel
        err = pred[idx] - truth
        ax2.plot(idx, err, '-', lw=0.8, alpha=0.6, label=name, color=c)

    ax1.set_ylabel('FAA (mGal)')
    ax1.set_title('沿 MR22-04 航迹的重力剖面对比 (每{}点采样)'.format(step))
    ax1.legend(loc='upper right', fontsize=9, ncol=2)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(idx[0], idx[-1])

    ax2.axhline(0, color='k', linestyle='--', lw=1)
    ax2.set_xlabel('验证点序号 (沿航迹时间顺序)')
    ax2.set_ylabel('误差 (mGal)')
    ax2.legend(loc='upper right', fontsize=8, ncol=3)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  航迹剖面: {save_path}")


def plot_validation_region_map(train_points, val_points, lon_range, lat_range, save_path):
    """验证区域地图: 显示训练巡航航迹与验证巡航 MR22-04 的位置."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot training cruises (gray, sparse)
    ax.scatter(train_points[:, 0], train_points[:, 1],
               s=0.5, c='gray', alpha=0.25, label=f'训练巡航 (30 个, n={len(train_points):,})')

    # Plot validation cruise MR22-04 (red, prominent)
    ax.scatter(val_points[:, 0], val_points[:, 1],
               s=0.8, c='red', alpha=0.6, label=f'验证巡航 MR22-04 (n={len(val_points):,})')

    # Region boundary
    rect_x = [lon_range[0], lon_range[1], lon_range[1], lon_range[0], lon_range[0]]
    rect_y = [lat_range[0], lat_range[0], lat_range[1], lat_range[1], lat_range[0]]
    ax.plot(rect_x, rect_y, 'b--', lw=1.5, alpha=0.7, label='验证区域')

    ax.set_xlim(lon_range[0] - 1, lon_range[1] + 1)
    ax.set_ylim(lat_range[0] - 1, lat_range[1] + 1)
    ax.set_xlabel('经度 (°E)')
    ax.set_ylabel('纬度 (°N)')
    ax.set_title(f'独立验证区域: 日本海沟及周边\n'
                 f'lon=[{lon_range[0]}°E, {lon_range[1]}°E], '
                 f'lat=[{lat_range[0]}°N, {lat_range[1]}°N]\n'
                 f'训练: 30 个巡航 (灰)  |  验证: MR22-04 (红, 完全独立)')
    ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle=':')
    ax.set_aspect('equal', adjustable='box')

    # Annotate key features
    ax.text(lon_range[0] + 0.5, lat_range[1] - 0.5,
            '验证区域特点:\n'
            '- 含日本海沟 (最深 ~10km)\n'
            '- 重力场变化剧烈\n'
            '- 含海陆边界\n'
            '- 是模型精度的严苛测试场',
            fontsize=9, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85),
            transform=ax.transData)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  验证区域地图: {save_path}")


def plot_spatial_error_map(val_points, predictions, method_names, save_path,
                            lon_range, lat_range):
    """误差空间分布: 每个验证点的误差 (模型 - 船载) 在地图上的颜色."""
    n = len(method_names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5),
                              sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    # Common color scale based on 99th percentile of |error| across all methods
    abs_errs = []
    for pred in predictions:
        valid = ~(np.isnan(pred) | np.isnan(val_points[:, 2]))
        abs_errs.append(np.abs(pred[valid] - val_points[valid, 2]))
    vmax = np.percentile(np.concatenate(abs_errs), 95)

    for ax, pred, name in zip(axes, predictions, method_names):
        valid = ~(np.isnan(pred) | np.isnan(val_points[:, 2]))
        err = pred[valid] - val_points[valid, 2]
        sc = ax.scatter(val_points[valid, 0], val_points[valid, 1],
                        c=err, s=1.5, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                        alpha=0.7)
        ax.set_xlim(lon_range)
        ax.set_ylim(lat_range)
        ax.set_xlabel('经度 (°E)')
        ax.set_ylabel('纬度 (°N)')
        rmse = np.sqrt(np.mean(err ** 2))
        ax.set_title(f'{name}\nRMSE={rmse:.2f} mGal  (n={valid.sum():,})')
        ax.grid(alpha=0.3, linestyle=':')
        ax.set_aspect('equal', adjustable='box')
        plt.colorbar(sc, ax=ax, label='误差 (mGal)', shrink=0.8)

    fig.suptitle('验证点误差空间分布 (模型 - 船载真值)\n'
                 '蓝 = 模型偏低, 红 = 模型偏高',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  误差空间分布: {save_path}")


def plot_method_diff_map(val_points, predictions, method_names, save_path,
                          lon_range, lat_range):
    """方法差异空间图: KF 与 Bilinear 在每个验证点的误差差异."""
    if 'KF' not in method_names or 'Bilinear' not in method_names:
        print(f"  跳过方法差异图 (需要 KF 和 Bilinear)")
        return

    kf_pred = predictions[method_names.index('KF')]
    bi_pred = predictions[method_names.index('Bilinear')]
    truth = val_points[:, 2]

    valid = ~(np.isnan(kf_pred) | np.isnan(bi_pred) | np.isnan(truth))
    err_kf = kf_pred[valid] - truth[valid]
    err_bi = bi_pred[valid] - truth[valid]
    diff = err_kf - err_bi  # < 0 means KF better

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                              gridspec_kw={'width_ratios': [5, 2]})
    ax_map, ax_hist = axes

    # Map: negative (green, KF better) vs positive (red, Bilinear better)
    vmax = np.percentile(np.abs(diff), 95)
    sc = ax_map.scatter(val_points[valid, 0], val_points[valid, 1],
                        c=diff, s=2, cmap='RdYlGn', vmin=-vmax, vmax=vmax,
                        alpha=0.75)
    ax_map.set_xlim(lon_range)
    ax_map.set_ylim(lat_range)
    ax_map.set_xlabel('经度 (°E)')
    ax_map.set_ylabel('纬度 (°N)')
    n_better = (diff < 0).sum()
    n_worse = (diff > 0).sum()
    ax_map.set_title(f'KF 相对 Bilinear 的逐点改进\n'
                     f'绿: KF 更准 ({n_better:,} 点, {n_better/len(diff)*100:.1f}%)  '
                     f'红: Bilinear 更准 ({n_worse:,} 点, {n_worse/len(diff)*100:.1f}%)')
    ax_map.grid(alpha=0.3, linestyle=':')
    ax_map.set_aspect('equal', adjustable='box')
    plt.colorbar(sc, ax=ax_map, label='err(KF) - err(Bilinear) (mGal)', shrink=0.8)

    # Histogram of the difference
    ax_hist.hist(diff, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
    ax_hist.axvline(0, color='k', linestyle='--', lw=1.5)
    ax_hist.axvline(diff.mean(), color='red', linestyle='-',
                    lw=1.5, label=f'均值={diff.mean():.2f}')
    ax_hist.set_xlabel('err(KF) - err(Bilinear) (mGal)')
    ax_hist.set_ylabel('点数')
    ax_hist.set_title('逐点改进分布')
    ax_hist.legend(fontsize=9)
    ax_hist.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  方法差异图: {save_path}")


def plot_qq_normal(val_truth, predictions, method_names, save_path):
    """Q-Q 图: 检验误差是否符合正态分布."""
    from scipy.stats import probplot
    n = len(method_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, pred, name in zip(axes, predictions, method_names):
        valid = ~(np.isnan(pred) | np.isnan(val_truth))
        diff = pred[valid] - val_truth[valid]
        diff_centered = diff - diff.mean()
        (osm, osr), (slope, intercept, r) = probplot(diff_centered, dist="norm")
        ax.scatter(osm, osr, s=3, alpha=0.4, c='steelblue')
        ax.plot(osm, slope * osm + intercept, 'r-', lw=1.5,
                label=f'线性拟合 R2={r**2:.4f}')
        std = diff_centered.std()
        ax.plot(osm, std * osm, 'k--', lw=1, alpha=0.6, label=f'N(0, {std:.2f}^2)')
        ax.set_xlabel('理论分位数 (标准正态)')
        ax.set_ylabel('样本分位数 (中心化误差, mGal)')
        ax.set_title(f'{name}\nSTD={std:.2f} mGal, 正态 R2={r**2:.4f}')
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(alpha=0.3)

    fig.suptitle('Q-Q 图: 误差正态性检验\n(点越贴近直线 → 误差越接近正态)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Q-Q 图: {save_path}")


def plot_error_vs_distance(train_points, val_points, predictions, method_names, save_path):
    """误差绝对值 vs 距最近训练点距离: 检验 KF 空间传播能力."""
    tree = cKDTree(train_points[:, :2])
    dist, _ = tree.query(val_points[:, :2], k=1)
    # Convert to km (approximate, 1° ~ 111 km)
    dist_km = dist * 111.0

    n = len(method_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, pred, name in zip(axes, predictions, method_names):
        valid = ~(np.isnan(pred) | np.isnan(val_points[:, 2]))
        err = np.abs(pred[valid] - val_points[valid, 2])
        d = dist_km[valid]

        # Bin by distance
        n_bins = 12
        d_max = np.percentile(d, 99)
        bins = np.linspace(0, d_max, n_bins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        rmse_per_bin = np.full(n_bins, np.nan)
        for i in range(n_bins):
            mask = (d >= bins[i]) & (d < bins[i + 1])
            if mask.sum() > 5:
                rmse_per_bin[i] = np.sqrt(np.mean(err[mask] ** 2))

        ax.scatter(d, err, s=1, alpha=0.15, c='lightgray')
        ax.plot(bin_centers, rmse_per_bin, 'o-', color='tab:red',
                lw=1.5, markersize=6, label='分段 RMSE')
        # Trend line through bin centers
        valid_bins = ~np.isnan(rmse_per_bin)
        if valid_bins.sum() > 1:
            ax.plot(bin_centers[valid_bins], rmse_per_bin[valid_bins],
                    '--', color='tab:blue', alpha=0.5, lw=1)

        ax.set_xlabel('距最近训练点距离 (km)')
        ax.set_ylabel('|误差| (mGal)' if ax is axes[0] else '')
        ax.set_title(f'{name}')
        ax.set_xlim(0, d_max)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle('误差 vs 距最近训练点距离\n(KF 应在远距离上保持低误差 → 体现空间传播能力)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  误差-距离图: {save_path}")


def plot_metrics_bar(val_truth, predictions, method_names, save_path):
    """多指标柱状图对比."""
    metrics = {name: {} for name in method_names}
    for name, pred in zip(method_names, predictions):
        valid = ~(np.isnan(pred) | np.isnan(val_truth))
        p, t = pred[valid], val_truth[valid]
        diff = p - t
        metrics[name]['RMSE'] = np.sqrt(np.mean(diff ** 2))
        metrics[name]['MAE'] = np.mean(np.abs(diff))
        metrics[name]['STD'] = np.std(diff)
        metrics[name]['Bias'] = np.abs(np.mean(diff))
        metrics[name]['Max|err|'] = np.percentile(np.abs(diff), 95)

    metric_names = ['RMSE', 'MAE', 'STD', 'Bias', 'Max|err|']
    n_metrics = len(metric_names)
    n_methods = len(method_names)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple', 'tab:orange']

    for ax, mname in zip(axes, metric_names):
        vals = [metrics[name][mname] for name in method_names]
        bars = ax.bar(range(n_methods), vals, color=colors[:n_methods],
                       edgecolor='black', linewidth=0.5, alpha=0.85)
        ax.set_xticks(range(n_methods))
        ax.set_xticklabels(method_names, rotation=15, fontsize=9)
        ax.set_ylabel(f'{mname} (mGal)')
        ax.set_title(mname)
        ax.grid(alpha=0.3, axis='y', linestyle=':')
        # Highlight best (min for all metrics)
        best_idx = int(np.argmin(vals))
        bars[best_idx].set_edgecolor('green')
        bars[best_idx].set_linewidth(2.5)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.02, f'{v:.2f}',
                    ha='center', va='bottom', fontsize=9,
                    fontweight='bold' if i == best_idx else 'normal')
        ax.set_ylim(0, max(vals) * 1.15)

    fig.suptitle(f'15'' 模型精度多指标对比 (n={len(val_truth):,} 验证点)\n'
                 f'绿框 = 该指标最优方法',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  指标柱状图: {save_path}")


def main(lon_range=(140, 150), lat_range=(30, 40), upscale=4):
    print("=" * 70)
    print("  15'' 模型 vs 船载重力真值 — 可视化对比")
    print("=" * 70)

    # Load data
    print("\n[1/3] Loading SSV + cruises...")
    gravity_lr, _, lon_lr, lat_lr = load_ssv_region(lon_range, lat_range, load_vgg=False)
    ny_lr, nx_lr = gravity_lr.shape
    ny_hr, nx_hr = ny_lr * upscale, nx_lr * upscale
    lon_hr = np.linspace(lon_range[0], lon_range[1], nx_hr)
    lat_hr = np.linspace(lat_range[0], lat_range[1], ny_hr)

    train_points, val_points = load_all_cruises_split(lon_range, lat_range, VAL_CRUISE)
    print(f"  Val: {len(val_points):,} points (MR22-04)")

    # Build training grid (SSV datum)
    ship_grav = assign_points_to_grid(train_points, lon_hr, lat_hr)
    train_mask_raw = ~np.isnan(ship_grav)
    ssv_hr = sp_zoom(gravity_lr, (upscale, upscale), order=1)
    diff_train_ssv = ship_grav[train_mask_raw] - ssv_hr[train_mask_raw]
    if abs(np.nanmean(diff_train_ssv)) >= 5 or np.nanstd(diff_train_ssv) >= 15:
        ssv_trend_lr = gaussian_filter(gravity_lr, sigma=10)
        ssv_trend_hr = sp_zoom(ssv_trend_lr, (upscale, upscale), order=1)
        ship_grav[train_mask_raw] = ship_grav[train_mask_raw] + ssv_trend_hr[train_mask_raw]
    train_mask = ~np.isnan(ship_grav)

    # === Build 3 models ===
    print("\n[2/3] Building 3 models...")
    models = {}

    # Bilinear (传统基线)
    models['Bilinear'] = ssv_hr.copy()

    # KF (本项目最优方法, 15" 网格)
    print("  KF (15'' grid)...")
    r = KalmanGravityReconstructor(
        grid_shape=(ny_hr, nx_hr),
        region_size_deg=lat_range[1] - lat_range[0],
        lr_grid_shape=(ny_lr, nx_lr))
    r.initialize(
        gravity_lr=gravity_lr, gravity_ship=ship_grav,
        ship_train_mask=train_mask,
        process_noise_std=2.0, sat_noise_std=3.0, ship_noise_std=1.0)
    models['KF'] = r.run(n_iterations=6, smooth_sigma=15.0)

    # Freq Fusion (传统方法对照)
    print("  FreqFusion...")
    ssv_hr_smooth = gaussian_filter(ssv_hr, sigma=2.0)
    ship_filled = ship_grav.copy()
    nan_mask = np.isnan(ship_filled)
    ship_filled[nan_mask] = ssv_hr[nan_mask]
    ship_filled = gaussian_filter(ship_filled, sigma=2.0)
    dx_deg = (lon_range[1] - lon_range[0]) / nx_hr
    dy_deg = (lat_range[1] - lat_range[0]) / ny_hr
    lat_center = (lat_range[0] + lat_range[1]) / 2.0
    ship_hp = butterworth_2d_filter(ship_filled, dx_deg, dy_deg, lat_center,
                                     27.0, btype='high', order=4)
    ssv_lp = butterworth_2d_filter(ssv_hr_smooth, dx_deg, dy_deg, lat_center,
                                    27.0, btype='low', order=4)
    models['FreqFusion'] = ssv_lp + ship_hp

    # === Get predictions at val points ===
    print("\n[3/3] Generating visualizations...")
    val_truth = val_points[:, 2]
    method_names = list(models.keys())
    predictions = [get_model_predictions_at_val(m, val_points, lon_hr, lat_hr)
                   for m in [models[n] for n in method_names]]

    # Print metrics table
    print("\n" + "=" * 70)
    print("  15'' 模型 vs 船载重力真值 (MR22-04, n={:,})".format(len(val_points)))
    print("=" * 70)
    print(f"  {'Method':<14} {'RMSE':<10} {'RMSE_bc':<10} {'STD':<8} {'Corr':<8} {'Bias':<10}")
    print("  " + "-" * 60)
    for name, pred in zip(method_names, predictions):
        valid = ~(np.isnan(pred) | np.isnan(val_truth))
        m = evaluate_15s_raw(models[name], val_points, lon_hr, lat_hr)
        print(f"  {name:<14} {m['RMSE']:<10.2f} {m['RMSE_bc']:<10.2f} "
              f"{m['STD']:<8.2f} {m['Correlation']:<8.4f} {m['Bias']:<10.2f}")

    # Generate plots
    print("\n  生成可视化图...")
    plot_validation_region_map(train_points, val_points, lon_range, lat_range,
                                OUTPUT_DIR / "val_region_map.png")
    plot_scatter(val_truth, predictions, method_names,
                 OUTPUT_DIR / "val_scatter_model_vs_ship.png")
    plot_error_hist(val_truth, predictions, method_names,
                    OUTPUT_DIR / "val_error_histogram.png")
    plot_track_profile(val_points, predictions, method_names,
                       OUTPUT_DIR / "val_track_profile.png")
    plot_spatial_error_map(val_points, predictions, method_names,
                            OUTPUT_DIR / "val_spatial_error_map.png",
                            lon_range, lat_range)
    plot_method_diff_map(val_points, predictions, method_names,
                          OUTPUT_DIR / "val_method_diff_map.png",
                          lon_range, lat_range)
    plot_qq_normal(val_truth, predictions, method_names,
                   OUTPUT_DIR / "val_qq_plot.png")
    plot_error_vs_distance(train_points, val_points, predictions, method_names,
                            OUTPUT_DIR / "val_error_vs_distance.png")
    plot_metrics_bar(val_truth, predictions, method_names,
                     OUTPUT_DIR / "val_metrics_bar.png")

    print(f"\n  所有图保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
