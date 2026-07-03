"""M2 α 参数扫描 — 20°×16° 大区域.

区域: lon=132-152°E, lat=26-42°N (与对比实验同区域)
α 候选: [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
- α=0 等价于纯 SSV 双线性 (基线对照)
- α=0.5 当前最优 (10°×10° 时确定)
- α>0.5 测试更强地形先验

输出:
  - output_real/m2_alpha_sweep_20x16.png  (α vs RMSE 折线图)
  - output_real/m2_alpha_sweep_20x16.txt  (数值表)
"""
import sys, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from independent_validation_v2 import VAL_CRUISE
from compare_m2_traditional import (
    load_region_data, get_predictions_at_val, compute_metrics)
from kf_novel import BathyPriorKF

OUTPUT_DIR = Path(__file__).parent.parent / "output_real"


def run_alpha_sweep(lon_range=(132, 152), lat_range=(26, 42), upscale=4,
                    alpha_list=(0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0),
                    val_cruise=VAL_CRUISE):
    print("=" * 70)
    print("  M2 α 参数扫描 — 20°×16° 大区域")
    print("=" * 70)
    t_start = time.time()

    print(f"\n[1/3] 加载数据 (区域 lon={lon_range}, lat={lat_range})...")
    data = load_region_data(lon_range, lat_range, val_cruise=val_cruise, upscale=upscale)
    data['lon_range'] = lon_range
    data['lat_range'] = lat_range

    val_truth = data['val_points'][:, 2]
    region_size = lat_range[1] - lat_range[0]

    print(f"\n[2/3] α 扫描 (共 {len(alpha_list)} 个 α)...")
    results = []
    for alpha in alpha_list:
        print(f"\n  --- α={alpha} ---")
        t0 = time.time()
        m2 = BathyPriorKF(
            hr_grid_shape=(data['ny_hr'], data['nx_hr']),
            lr_grid_shape=(data['ny_lr'], data['nx_lr']),
            region_size_deg=region_size, alpha=alpha)
        gravity_15s = m2.build(
            gravity_lr=data['gravity_lr'], ship_15s=data['ship_grav'],
            train_mask=data['train_mask'],
            lon_hr=data['lon_hr'], lat_hr=data['lat_hr'],
            lon_range=lon_range, lat_range=lat_range,
            upscale=upscale, bathy_15s=data['bathy_15s'],
            n_iter=6, smooth_sigma=15.0)
        elapsed = time.time() - t0

        pred = get_predictions_at_val(gravity_15s, data['val_points'],
                                       data['lon_hr'], data['lat_hr'])
        m = compute_metrics(pred, val_truth)
        m['alpha'] = alpha
        m['time'] = elapsed
        results.append(m)
        print(f"  α={alpha:.1f}: RMSE={m['RMSE']:.3f}  MAE={m['MAE']:.3f}  "
              f"Corr={m['Corr']:.4f}  Bias={m['Bias']:+.2f}  ({elapsed:.1f}s)")

    best = min(results, key=lambda r: r['RMSE'])
    print(f"\n  最优 α={best['alpha']}  RMSE={best['RMSE']:.3f}")

    print(f"\n[3/3] 生成图与表...")
    save_png = OUTPUT_DIR / "m2_alpha_sweep_20x16.png"
    save_txt = OUTPUT_DIR / "m2_alpha_sweep_20x16.txt"

    fig, ax = plt.subplots(figsize=(10, 6))
    alphas = [r['alpha'] for r in results]
    rmses = [r['RMSE'] for r in results]
    maes = [r['MAE'] for r in results]
    stds = [r['STD'] for r in results]

    ax.plot(alphas, rmses, 'o-', color='#D7301F', lw=2, ms=10, label='RMSE')
    ax.plot(alphas, maes, 's--', color='#1A9850', lw=1.5, ms=8, label='MAE')
    ax.plot(alphas, stds, '^:', color='#4575B4', lw=1.5, ms=8, label='STD')

    best_idx = rmses.index(best['RMSE'])
    ax.axvline(best['alpha'], color='gray', ls=':', lw=1, alpha=0.7)
    ax.annotate(f'最优 α={best["alpha"]}\nRMSE={best["RMSE"]:.3f}',
                xy=(best['alpha'], best['RMSE']),
                xytext=(best['alpha'] + 0.15, best['RMSE'] + 0.15),
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black'),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF7BC',
                          edgecolor='#FDB863', alpha=0.95))

    ax.axhline(rmses[0], color='gray', ls='--', lw=0.8, alpha=0.5,
               label=f'α=0 基线 RMSE={rmses[0]:.3f}')

    ax.set_xlabel('α (地形先验权重)', fontsize=12)
    ax.set_ylabel('误差 (mGal)', fontsize=12)
    ax.set_title(f'M2 α 参数扫描 — 20°×16° 大区域\n'
                 f'验证: {val_cruise} (n={len(data["val_points"]):,})',
                 fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(alphas)
    plt.tight_layout()
    plt.savefig(save_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  折线图: {save_png}")

    with open(save_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"  M2 α 参数扫描 — 20°×16° 大区域\n")
        f.write(f"  区域: lon={lon_range}, lat={lat_range}\n")
        f.write(f"  验证: {val_cruise} (n={len(data['val_points']):,})\n")
        f.write("=" * 80 + "\n")
        f.write(f"  {'α':<6} {'RMSE':<10} {'MAE':<10} {'Max|e|':<10} "
                f"{'STD':<10} {'Corr':<10} {'Bias':<10} {'Time(s)':<8}\n")
        f.write("-" * 80 + "\n")
        for r in results:
            f.write(f"  {r['alpha']:<6.2f} {r['RMSE']:<10.3f} {r['MAE']:<10.3f} "
                    f"{r['Max|e|']:<10.2f} {r['STD']:<10.3f} {r['Corr']:<10.4f} "
                    f"{r['Bias']:<+10.2f} {r['time']:<8.1f}\n")
        f.write("=" * 80 + "\n")
        f.write(f"  最优: α={best['alpha']}  RMSE={best['RMSE']:.3f}\n")
    print(f"  指标表: {save_txt}")

    print(f"\n  总用时: {time.time()-t_start:.0f}s")
    return results, best


if __name__ == "__main__":
    run_alpha_sweep()
