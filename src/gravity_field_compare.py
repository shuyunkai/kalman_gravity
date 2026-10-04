"""3 方法验证区域重力异常场对比图.

在 20°×16° 验证区域 (lon=132-152, lat=26-42) 上构建 3 方法 (双线性/频域融合/KF)
的 15'' 重力异常场, 生成对比图:
  - 上: 3 子图 (3 方法的重力异常场, 色码一致, 黑线=MR22-04 航迹, 黄点=训练点)
  - 下: 差异场 (KF - 双线性 / KF - 频域融合 / 频域融合 - 双线性), 显示方法间结构差异
"""
import sys, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).parent))

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from independent_validation_v2 import VAL_CRUISE
from three_method_compare import (
    load_region_data, build_three_methods,
    METHOD_BILINEAR, METHOD_FREQFUSION, METHOD_KF)

OUTPUT_DIR = Path(__file__).parent.parent / "output_real"


def main(lon_range=(132, 152), lat_range=(26, 42), alpha=0.2):
    print("=" * 70)
    print("  3 方法验证区域重力异常场对比")
    print("=" * 70)
    t_start = time.time()

    print(f"\n[1/2] 加载数据 + 构建 3 方法 (lon={lon_range}, lat={lat_range})...")
    data = load_region_data(lon_range, lat_range, val_cruise=VAL_CRUISE, upscale=4)
    data['lon_range'] = lon_range
    data['lat_range'] = lat_range
    models, times = build_three_methods(data, alpha=alpha)

    print(f"\n[2/2] 生成重力异常场对比图...")
    _plot_gravity_fields(models, data, lon_range, lat_range, alpha, times)

    print(f"\n  总用时: {time.time()-t_start:.0f}s")


def _plot_gravity_fields(models, data, lon_range, lat_range, alpha, times):
    lon_hr, lat_hr = data['lon_hr'], data['lat_hr']
    val_points = data['val_points']
    train_points = data['train_points']

    method_names = [METHOD_BILINEAR, METHOD_FREQFUSION, METHOD_KF]
    extent = [lon_hr[0], lon_hr[-1], lat_hr[0], lat_hr[-1]]

    # 统一色码 (基于 3 方法所有值的 0.5/99.5 百分位)
    all_vals = np.concatenate([models[n].ravel() for n in method_names])
    all_vals = all_vals[~np.isnan(all_vals)]
    vmax = float(np.percentile(np.abs(all_vals), 99.5))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharex=False, sharey=False)

    # 经纬度刻度: 从区域起止点开始, 间隔均匀
    lon_ticks = np.arange(lon_range[0], lon_range[1] + 1, 5)
    if lon_ticks[-1] != lon_range[1]:
        lon_ticks = np.append(lon_ticks, lon_range[1])
    lat_ticks = np.arange(lat_range[0], lat_range[1] + 1, 4)
    if lat_ticks[-1] != lat_range[1]:
        lat_ticks = np.append(lat_ticks, lat_range[1])

    def _setup_ax(ax):
        ax.set_xticks(lon_ticks)
        ax.set_yticks(lat_ticks)
        ax.set_xticklabels([f'{t:.0f}°E' for t in lon_ticks], fontsize=9)
        ax.set_yticklabels([f'{t:.0f}°N' for t in lat_ticks], fontsize=9)
        ax.set_xlabel('经度', fontsize=10)
        ax.set_ylabel('纬度', fontsize=10)
        ax.grid(True, color='gray', lw=0.4, alpha=0.35, linestyle='--')
        ax.tick_params(labelsize=9)

    # === 上排: 3 方法重力异常场 ===
    for ax, name in zip(axes[0], method_names):
        im = ax.imshow(models[name], cmap='RdBu_r', norm=norm, aspect='equal',
                       origin='lower', extent=extent, interpolation='nearest')
        ax.plot(val_points[:, 0], val_points[:, 1], 'k-', lw=0.3, alpha=0.5)
        ax.scatter(train_points[:, 0], train_points[:, 1],
                   s=0.15, c='yellow', alpha=0.2, edgecolors='none')
        ax.set_xlim(lon_range)
        ax.set_ylim(lat_range)
        _setup_ax(ax)
        ax.set_title(f'{name}', fontsize=13,
                     fontweight='bold' if name == METHOD_KF else 'normal')

    # === 下排: 差异场 ===
    diffs = [
        ('KF - 双线性',   models[METHOD_KF] - models[METHOD_BILINEAR]),
        ('KF - 频域融合', models[METHOD_KF] - models[METHOD_FREQFUSION]),
        ('频域融合 - 双线性', models[METHOD_FREQFUSION] - models[METHOD_BILINEAR]),
    ]

    all_diff = np.concatenate([d.ravel() for _, d in diffs])
    all_diff = all_diff[~np.isnan(all_diff)]
    vmax_diff = float(np.percentile(np.abs(all_diff), 99))
    norm_diff = TwoSlopeNorm(vmin=-vmax_diff, vcenter=0, vmax=vmax_diff)

    for ax, (label, diff) in zip(axes[1], diffs):
        im_d = ax.imshow(diff, cmap='RdBu_r', norm=norm_diff, aspect='equal',
                         origin='lower', extent=extent, interpolation='nearest')
        ax.plot(val_points[:, 0], val_points[:, 1], 'k-', lw=0.3, alpha=0.5)
        ax.set_xlim(lon_range)
        ax.set_ylim(lat_range)
        _setup_ax(ax)
        sigma = float(np.std(diff[~np.isnan(diff)]))
        max_abs = float(np.nanmax(np.abs(diff)))
        ax.set_title(f'{label}\nσ={sigma:.2f} mGal, max|Δ|={max_abs:.1f}',
                     fontsize=12)

    # 色标
    cb1 = fig.colorbar(im, ax=axes[0].tolist(), orientation='vertical',
                       pad=0.015, shrink=0.95, aspect=30, extend='both')
    cb1.set_label('重力异常 FAA (mGal)', fontsize=11)

    cb2 = fig.colorbar(im_d, ax=axes[1].tolist(), orientation='vertical',
                       pad=0.015, shrink=0.95, aspect=30, extend='both')
    cb2.set_label('重力差异 (mGal)', fontsize=11)

    # 区域说明
    region_desc = (f'验证区域: lon={lon_range[0]}°-{lon_range[1]}°E, '
                   f'lat={lat_range[0]}°-{lat_range[1]}°N  '
                   f'(20°×16°, 日本本州以南-日本海沟-伊豆-小笠原北部, '
                   f'含西北太平洋与日本海东南部)')

    fig.suptitle(f'3 方法验证区域重力异常场对比 (α={alpha})\n'
                 f'上: 3 方法重力异常场 (黑线=MR22-04 航迹, 黄点=训练点)  '
                 f'下: 差异场 (方法间结构差异)\n'
                 f'{region_desc}',
                 fontsize=12.5, fontweight='bold', y=0.995)

    save_png = OUTPUT_DIR / "fig_gravity_field_compare.png"
    plt.savefig(save_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {save_png}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.2)
    args = parser.parse_args()
    main(alpha=args.alpha)
