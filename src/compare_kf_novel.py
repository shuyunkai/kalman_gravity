"""四种 KF 新方法 vs 双线性基线 — 小区域试验 + 独立验证.

测试区域: 日本海沟 lon=142-146°E, lat=34-38°N (4°×4°)
- 1' 网格: 240×240
- 15" 网格: 960×960
- 训练: 30 个日本巡航 (排除 MR22-04)
- 验证: MR22-04 独立巡航

输出:
  - output_real/kf_novel_comparison.png  — 5 个面板 (双线性 + 4 种新方法)
  - output_real/kf_novel_metrics.txt     — 数值对比表
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
    load_all_cruises_split, assign_points_to_grid, VAL_CRUISE,
    evaluate_15s_raw)
from run_pipeline import load_ssv_region, load_gebco_region
from kf_novel import (
    MultiScaleResidualKF, BathyPriorKF, AlongTrackKF, WaveletKF)

OUTPUT_DIR = Path(__file__).parent.parent / "output_real"


def get_predictions_at_val(gravity_15s, val_points, lon_hr, lat_hr):
    interp = RegularGridInterpolator(
        (lat_hr, lon_hr), gravity_15s, bounds_error=False, fill_value=np.nan)
    return interp(np.column_stack([val_points[:, 1], val_points[:, 0]]))


def load_bathy_15s(lon_range, lat_range, lon_hr, lat_hr):
    """加载 GEBCO 15" 水深并插值到目标 15" 网格."""
    bathy_lr, lon_b, lat_b = load_gebco_region(lon_range, lat_range)
    # GEBCO 已经是 15", 但 lon_b/lat_b 可能不完全对齐, 用插值
    interp = RegularGridInterpolator(
        (lat_b, lon_b), bathy_lr, bounds_error=False, fill_value=0)
    lon_mesh, lat_mesh = np.meshgrid(lon_hr, lat_hr)
    bathy_15s = interp(np.column_stack([lat_mesh.ravel(), lon_mesh.ravel()])).reshape(
        len(lat_hr), len(lon_hr))
    return bathy_15s


def main(lon_range=(142, 146), lat_range=(34, 38), upscale=4):
    print("=" * 70)
    print("  四种 KF 新方法 vs 双线性 — 小区域对比试验")
    print("=" * 70)
    t_start = time.time()

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

    # 船载训练点赋值到 15" 网格 + SSV 基准对齐
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

    # 加载 GEBCO 水深 (方法2 需要)
    print("  加载 GEBCO 15'' 水深...")
    bathy_15s = load_bathy_15s(lon_range, lat_range, lon_hr, lat_hr)
    print(f"  GEBCO 水深: shape={bathy_15s.shape}, "
          f"range=[{bathy_15s.min():.0f}, {bathy_15s.max():.0f}] m")

    region_size = lat_range[1] - lat_range[0]

    # === 构建所有方法 ===
    print("\n[2/4] 构建 5 个模型 (双线性基线 + 4 种 KF 新方法)...")
    models = {}
    times = {}

    # Bilinear baseline
    t0 = time.time()
    models['双线性插值'] = ssv_hr.copy()
    times['双线性插值'] = time.time() - t0
    print(f"  双线性插值: {times['双线性插值']:.1f}s")

    # Method 1: Multi-scale Residual KF
    t0 = time.time()
    print("\n  --- 方法 1: 多尺度残差 KF ---")
    m1 = MultiScaleResidualKF(
        lr_grid_shape=(ny_lr, nx_lr), hr_grid_shape=(ny_hr, nx_hr),
        region_size_deg=region_size)
    models['M1 多尺度残差KF'] = m1.build(
        gravity_lr=gravity_lr, ship_15s=ship_grav, train_mask=train_mask,
        lon_hr=lon_hr, lat_hr=lat_hr, lon_range=lon_range, lat_range=lat_range,
        upscale=upscale)
    times['M1 多尺度残差KF'] = time.time() - t0
    print(f"  M1 完成: {times['M1 多尺度残差KF']:.1f}s")

    # Method 2: Bathy Prior KF
    t0 = time.time()
    print("\n  --- 方法 2: 地形物理先验 KF ---")
    m2 = BathyPriorKF(
        hr_grid_shape=(ny_hr, nx_hr), lr_grid_shape=(ny_lr, nx_lr),
        region_size_deg=region_size, alpha=1.0)
    models['M2 地形先验KF'] = m2.build(
        gravity_lr=gravity_lr, ship_15s=ship_grav, train_mask=train_mask,
        lon_hr=lon_hr, lat_hr=lat_hr, lon_range=lon_range, lat_range=lat_range,
        upscale=upscale, bathy_15s=bathy_15s)
    times['M2 地形先验KF'] = time.time() - t0
    print(f"  M2 完成: {times['M2 地形先验KF']:.1f}s")

    # Method 3: Along-Track KF
    t0 = time.time()
    print("\n  --- 方法 3: 沿迹观测 KF ---")
    # 沿迹方法需要原始点序列 (含 MR22-04 之外的所有训练点)
    # train_points 已经是 (lon, lat, faa) 数组
    # 构造 train_mask_pts: True 表示训练点 (这里全部训练点都参与)
    train_mask_pts = np.ones(len(train_points), dtype=bool)
    m3 = AlongTrackKF(
        hr_grid_shape=(ny_hr, nx_hr), lr_grid_shape=(ny_lr, nx_lr),
        region_size_deg=region_size)
    models['M3 沿迹观测KF'] = m3.build(
        gravity_lr=gravity_lr, ship_points_raw=train_points,
        train_mask_pts=train_mask_pts, lon_hr=lon_hr, lat_hr=lat_hr,
        lon_range=lon_range, lat_range=lat_range, upscale=upscale)
    times['M3 沿迹观测KF'] = time.time() - t0
    print(f"  M3 完成: {times['M3 沿迹观测KF']:.1f}s")

    # Method 4: Wavelet KF
    t0 = time.time()
    print("\n  --- 方法 4: 谱域小波 KF ---")
    m4 = WaveletKF(
        hr_grid_shape=(ny_hr, nx_hr), lr_grid_shape=(ny_lr, nx_lr),
        region_size_deg=region_size, n_levels=3)
    models['M4 谱域小波KF'] = m4.build(
        gravity_lr=gravity_lr, ship_15s=ship_grav, train_mask=train_mask,
        lon_hr=lon_hr, lat_hr=lat_hr, lon_range=lon_range, lat_range=lat_range,
        upscale=upscale)
    times['M4 谱域小波KF'] = time.time() - t0
    print(f"  M4 完成: {times['M4 谱域小波KF']:.1f}s")

    # === 评估 ===
    print("\n[3/4] 独立验证 (MR22-04, n={:,})...".format(len(val_points)))
    method_names = list(models.keys())
    val_truth = val_points[:, 2]
    predictions = [get_predictions_at_val(models[n], val_points, lon_hr, lat_hr)
                   for n in method_names]

    print("\n  " + "=" * 80)
    print(f"  {'Method':<22} {'RMSE':<10} {'MAE':<10} {'Max|e|':<10} "
          f"{'Corr':<10} {'Bias':<10} {'Time(s)':<8}")
    print("  " + "-" * 80)
    metrics_rows = []
    for name, pred in zip(method_names, predictions):
        valid = ~(np.isnan(pred) | np.isnan(val_truth))
        diff = pred[valid] - val_truth[valid]
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mae = float(np.mean(np.abs(diff)))
        maxe = float(np.max(np.abs(diff)))
        std = float(np.std(diff))
        corr = float(np.corrcoef(pred[valid], val_truth[valid])[0, 1])
        bias = float(np.mean(diff))
        metrics_rows.append((name, rmse, mae, maxe, std, corr, bias, times[name]))
        print(f"  {name:<22} {rmse:<10.2f} {mae:<10.2f} {maxe:<10.2f} "
              f"{corr:<10.4f} {bias:<10.2f} {times[name]:<8.1f}")

    # === 可视化 ===
    print("\n[4/4] 生成对比图...")
    fig, axes = plt.subplots(1, len(method_names), figsize=(5 * len(method_names), 5),
                              sharex=True, sharey=True)
    if len(method_names) == 1:
        axes = [axes]

    all_vals = np.concatenate([m.ravel() for m in models.values()])
    all_vals = all_vals[~np.isnan(all_vals)]
    vmax = float(np.percentile(np.abs(all_vals), 99.5))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    extent = [lon_hr[0], lon_hr[-1], lat_hr[0], lat_hr[-1]]

    for ax, name in zip(axes, method_names):
        im = ax.imshow(models[name], cmap='RdBu_r', norm=norm, aspect='equal',
                       origin='lower', extent=extent, interpolation='nearest')
        # 验证航迹
        ax.plot(val_points[:, 0], val_points[:, 1], 'k-', lw=0.4, alpha=0.55)
        # 训练点 (黄色稀疏)
        ax.scatter(train_points[:, 0], train_points[:, 1],
                   s=0.3, c='yellow', alpha=0.25, edgecolors='none')
        ax.set_xlim(lon_range)
        ax.set_ylim(lat_range)
        ax.set_xlabel('经度 (°E)', fontsize=9)
        ax.set_ylabel('纬度 (°N)', fontsize=9)
        # 找该方法的 RMSE
        row = next(r for r in metrics_rows if r[0] == name)
        ax.set_title(f'{name}\nRMSE={row[1]:.2f} mGal  ({times[name]:.1f}s)',
                     fontsize=10, fontweight='bold')
        ax.tick_params(labelsize=8)

    cb = fig.colorbar(im, ax=axes, orientation='vertical',
                      pad=0.015, shrink=0.85, aspect=35, extend='both')
    cb.set_label('FAA (mGal)', fontsize=10)
    fig.suptitle('四种 KF 新方法 vs 双线性基线 — 日本海沟小区域 (4°×4°) 对比\n'
                 '黑线=MR22-04 验证航迹, 黄点=训练点',
                 fontsize=11, y=1.02)
    out_png = OUTPUT_DIR / "kf_novel_comparison.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  对比图: {out_png}")

    # 保存 metrics 文本
    out_txt = OUTPUT_DIR / "kf_novel_metrics.txt"
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("  四种 KF 新方法 vs 双线性基线 (日本海沟 4°×4° 小区域)\n")
        f.write(f"  区域: lon={lon_range}, lat={lat_range}\n")
        f.write(f"  验证: MR22-04 (n={len(val_points):,})\n")
        f.write(f"  训练点: {len(train_points):,} (15'' 网格 {train_mask.sum():,} 像素)\n")
        f.write("=" * 80 + "\n")
        f.write(f"  {'Method':<22} {'RMSE':<10} {'MAE':<10} {'Max|e|':<10} "
                f"{'STD':<10} {'Corr':<10} {'Bias':<10} {'Time(s)':<8}\n")
        f.write("-" * 80 + "\n")
        for row in metrics_rows:
            f.write(f"  {row[0]:<22} {row[1]:<10.2f} {row[2]:<10.2f} {row[3]:<10.2f} "
                    f"{row[4]:<10.2f} {row[5]:<10.4f} {row[6]:<10.2f} {row[7]:<8.1f}\n")
    print(f"  指标表: {out_txt}")

    print(f"\n  总用时: {time.time()-t_start:.0f}s")
    return metrics_rows


if __name__ == "__main__":
    main()
