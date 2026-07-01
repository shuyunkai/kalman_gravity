"""15" 模型 vs 船载重力真值 — 可视化对比.

对比机制说明:
  - 船载重力 (MR22-04 巡航) = 真值 (船载重力仪精度 ~1-2 mGal)
  - 15" 模型网格 = 待评估对象
  - 对比方式: 在每个船载验证点 (lon, lat, faa_ship) 处,
    从 15" 模型网格双线性插值取模型值 faa_model,
    计算 diff = faa_model - faa_ship
  - RMSE = sqrt(mean(diff²)) = 模型相对船载真值的精度

输出:
  1. 散点图: 模型值 (y) vs 船载真值 (x), 理想情况 y=x
  2. 误差直方图: diff 分布, 理想情况以 0 为中心, 窄峰
  3. 沿航迹剖面: 验证点序列上模型与船载的对比
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
    plot_scatter(val_truth, predictions, method_names,
                 OUTPUT_DIR / "val_scatter_model_vs_ship.png")
    plot_error_hist(val_truth, predictions, method_names,
                    OUTPUT_DIR / "val_error_histogram.png")
    plot_track_profile(val_points, predictions, method_names,
                       OUTPUT_DIR / "val_track_profile.png")

    print(f"\n  所有图保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
