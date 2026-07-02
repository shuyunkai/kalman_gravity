"""日本海沟区域 真正 15'' KF 重力场图.

与全球图(1' KF 双线性显示)不同, 本图是真正的 15'' KF 重建:
  - 1' SSV33.1 卫星测高 → 双线性上采样到 15''
  - 30 个日本巡航 10s 原始点 → 精确像素赋值到 15'' 网格
  - 卡尔曼滤波融合多源数据, 状态变量在 15'' 网格上迭代更新
  - 输出: 2400 × 2400 = 5.76M 像素 的真正 15'' KF 重力场

时间: KF 重建 ~25s, 绘图 ~5s, 共 ~30s
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
from run_pipeline import load_ssv_region
from kalman_gravity import KalmanGravityReconstructor

OUTPUT_DIR = Path(__file__).parent.parent / "output_real"


def main(lon_range=(140, 150), lat_range=(30, 40), upscale=4):
    print("=" * 70)
    print("  日本海沟区域 真正 15'' KF 重力场重建与可视化")
    print("=" * 70)
    t_start = time.time()

    print("\n[1/4] 加载 SSV33.1 + 30 个日本巡航...")
    gravity_lr, _, lon_lr, lat_lr = load_ssv_region(lon_range, lat_range, load_vgg=False)
    ny_lr, nx_lr = gravity_lr.shape
    ny_hr, nx_hr = ny_lr * upscale, nx_lr * upscale
    lon_hr = np.linspace(lon_range[0], lon_range[1], nx_hr)
    lat_hr = np.linspace(lat_range[0], lat_range[1], ny_hr)
    print(f"  1' SSV: {gravity_lr.shape}  15'' 目标: ({ny_hr}, {nx_hr})")

    train_points, val_points = load_all_cruises_split(lon_range, lat_range, VAL_CRUISE)
    print(f"  训练: {len(train_points):,} 点  验证: {len(val_points):,} 点")

    print("\n[2/4] 训练点精确像素赋值 + 基准对齐...")
    ship_grav = assign_points_to_grid(train_points, lon_hr, lat_hr)
    train_mask_raw = ~np.isnan(ship_grav)
    ssv_hr = sp_zoom(gravity_lr, (upscale, upscale), order=1)
    diff_train_ssv = ship_grav[train_mask_raw] - ssv_hr[train_mask_raw]
    if abs(np.nanmean(diff_train_ssv)) >= 5 or np.nanstd(diff_train_ssv) >= 15:
        ssv_trend_lr = gaussian_filter(gravity_lr, sigma=10)
        ssv_trend_hr = sp_zoom(ssv_trend_lr, (upscale, upscale), order=1)
        ship_grav[train_mask_raw] = ship_grav[train_mask_raw] + ssv_trend_hr[train_mask_raw]
    train_mask = ~np.isnan(ship_grav)
    print(f"  15'' 网格训练点: {train_mask.sum():,} ({train_mask.sum()/train_mask.size*100:.2f}% 覆盖)")

    print("\n[3/4] 卡尔曼滤波 15'' 重建 (6 次迭代, smooth_sigma=15)...")
    t0 = time.time()
    r = KalmanGravityReconstructor(
        grid_shape=(ny_hr, nx_hr),
        region_size_deg=lat_range[1] - lat_range[0],
        lr_grid_shape=(ny_lr, nx_lr))
    r.initialize(
        gravity_lr=gravity_lr, gravity_ship=ship_grav,
        ship_train_mask=train_mask,
        process_noise_std=2.0, sat_noise_std=3.0, ship_noise_std=1.0)
    gravity_kf_15s = r.run(n_iterations=6, smooth_sigma=15.0)
    print(f"  KF 完成, 用时 {time.time()-t0:.0f}s")

    print("\n[4/4] 绘制真正 15'' KF 重力场图...")
    t0 = time.time()
    vmax = np.nanpercentile(np.abs(gravity_kf_15s), 99.5)
    print(f"  色标范围: ±{vmax:.1f} mGal (99.5 分位)")

    fig, ax = plt.subplots(figsize=(11, 9))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    extent = [lon_range[0], lon_range[1], lat_range[0], lat_range[1]]

    im = ax.imshow(gravity_kf_15s, cmap='RdBu_r', norm=norm, aspect='equal',
                   origin='lower', extent=extent, interpolation='nearest')

    cb = plt.colorbar(im, ax=ax, orientation='vertical',
                      pad=0.02, shrink=0.85, aspect=35, extend='both')
    cb.set_label('自由空气重力异常 FAA (mGal)', fontsize=11)
    cb.ax.tick_params(labelsize=9)

    # Overlay: 训练点(黄) + 验证航次 MR22-04(黑)
    ax.scatter(train_points[:, 0], train_points[:, 1],
               s=0.4, c='yellow', alpha=0.35, edgecolors='none',
               label=f'训练点 ({len(train_points):,})')
    ax.scatter(val_points[:, 0], val_points[:, 1],
               s=0.6, c='black', alpha=0.7, edgecolors='none',
               label=f'验证航次 MR22-04 ({len(val_points):,})')

    ax.set_xlim(lon_range)
    ax.set_ylim(lat_range)
    ax.set_xlabel('经度 (°E)', fontsize=11)
    ax.set_ylabel('纬度 (°N)', fontsize=11)
    ax.set_title(f'日本海沟区域 真正 15'' 卡尔曼滤波重力场模型\n'
                 f'KF 融合: 1\' SSV33.1 + 30 个日本巡航 10s 船载  '
                 f'(2400×2400 = 5.76M 像素)',
                 fontsize=11, pad=10)

    ax.set_xticks(np.arange(lon_range[0], lon_range[1]+0.1, 2))
    ax.set_yticks(np.arange(lat_range[0], lat_range[1]+0.1, 2))
    ax.grid(alpha=0.4, color='gray', linestyle=':', linewidth=0.5)
    ax.tick_params(labelsize=9)
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9)

    # 标注日本海沟
    ax.annotate('日本海沟\n(Japan Trench)',
                xy=(142.5, 36.5), xytext=(145, 32.5),
                fontsize=10, ha='center', color='darkred', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='darkred', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='darkred', alpha=0.9))

    stats_text = (
        f"统计信息:\n"
        f"  均值: {np.nanmean(gravity_kf_15s):.2f} mGal\n"
        f"  标准差: {np.nanstd(gravity_kf_15s):.2f} mGal\n"
        f"  范围: [{np.nanmin(gravity_kf_15s):.1f}, {np.nanmax(gravity_kf_15s):.1f}]\n"
        f"  分辨率: 真正 15\'\' ({ny_hr}×{nx_hr})\n"
        f"  KF 迭代: 6 次, smooth_sigma=15\n"
        f"  训练点覆盖: {train_mask.sum()/train_mask.size*100:.2f}%"
    )
    ax.text(0.013, 0.985, stats_text, transform=ax.transAxes,
            fontsize=8.5, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='gray', alpha=0.92))

    plt.tight_layout()
    out_path = OUTPUT_DIR / "japan_kf_gravity_15s.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  绘图完成, 用时 {time.time()-t0:.0f}s")
    print(f"  输出: {out_path} ({out_path.stat().st_size/1024**2:.1f} MB)")

    print(f"\n  总用时: {time.time()-t_start:.0f}s")
    return str(out_path)


if __name__ == "__main__":
    main()
