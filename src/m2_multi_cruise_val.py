"""M2 多航次独立验证 — 5 条 MR/KS 系列航次.

每条航次轮流做验证集, 其余作训练, 在 20°×16° 区域上跑 3 方法 (双线性/频域融合/M2):
  - MR22-04_gravity_dmo  (日本海沟, 主验证)
  - MR22-02_gravity_dmo
  - MR23-02_gravity_dmo
  - MR21-02_gravity_dmo
  - KS-21-27_gravity

输出:
  - output_real/m2_multi_cruise_metrics.txt  (5×3 表格)
  - output_real/m2_multi_cruise.png          (分组柱状图)
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

from compare_m2_traditional import (
    load_region_data, build_three_methods, get_predictions_at_val, compute_metrics)

OUTPUT_DIR = Path(__file__).parent.parent / "output_real"

VAL_CRUISES = [
    "MR22-04_gravity_dmo",
    "MR22-02_gravity_dmo",
    "MR23-02_gravity_dmo",
    "MR21-02_gravity_dmo",
    "KS-21-27_gravity",
]


def run_multi_cruise_val(lon_range=(132, 152), lat_range=(26, 42),
                         upscale=4, alpha=0.5):
    print("=" * 70)
    print("  M2 多航次独立验证 — 5 条 MR/KS 系列航次")
    print("=" * 70)
    t_start = time.time()

    all_results = {}  # {cruise: [(method, metrics), ...]}

    for i, cruise in enumerate(VAL_CRUISES):
        print(f"\n{'='*70}")
        print(f"  [{i+1}/{len(VAL_CRUISES)}] 验证航次: {cruise}")
        print(f"{'='*70}")

        try:
            data = load_region_data(lon_range, lat_range, val_cruise=cruise,
                                    upscale=upscale)
            data['lon_range'] = lon_range
            data['lat_range'] = lat_range
        except Exception as e:
            print(f"  跳过 {cruise}: 数据加载失败 {e}")
            all_results[cruise] = None
            continue

        if len(data['val_points']) < 100:
            print(f"  跳过 {cruise}: 验证点过少 (n={len(data['val_points'])})")
            all_results[cruise] = None
            continue

        print(f"  验证点: {len(data['val_points']):,}  训练点: {len(data['train_points']):,}")

        # 3 方法
        models, times = build_three_methods(data, alpha=alpha)

        val_truth = data['val_points'][:, 2]
        cruise_results = []
        for name in models:
            pred = get_predictions_at_val(models[name], data['val_points'],
                                          data['lon_hr'], data['lat_hr'])
            m = compute_metrics(pred, val_truth)
            m['time'] = times[name]
            cruise_results.append((name, m))
            print(f"  {name:<22} RMSE={m['RMSE']:.3f}  MAE={m['MAE']:.3f}  "
                  f"Corr={m['Corr']:.4f}  Bias={m['Bias']:+.2f}")

        all_results[cruise] = cruise_results
        print(f"  用时: {sum(times.values()):.0f}s")

    # === 汇总 ===
    print(f"\n\n{'='*70}")
    print("  多航次验证汇总")
    print(f"{'='*70}")

    valid_cruises = [c for c, r in all_results.items() if r is not None]
    if not valid_cruises:
        print("  ERROR: 无有效航次")
        return None

    method_names = [n for n, _ in all_results[valid_cruises[0]]]
    print(f"\n  {'航次':<25}", end="")
    for n in method_names:
        print(f"  {n:<22}", end="")
    print()
    print("  " + "-" * 95)
    for cruise in valid_cruises:
        cruise_short = cruise.replace("_gravity_dmo", "").replace("_gravity", "")
        print(f"  {cruise_short:<25}", end="")
        for name, m in all_results[cruise]:
            print(f"  RMSE={m['RMSE']:.3f} (n={m['n']})", end="  ")
        print()

    # 平均改善
    print(f"\n  各方法平均 RMSE:")
    avg_rmse = {}
    for name in method_names:
        rmses = [next(m for n, m in all_results[c] if n == name)['RMSE']
                 for c in valid_cruises]
        avg_rmse[name] = np.mean(rmses)
        print(f"    {name:<22}: {avg_rmse[name]:.3f} mGal  "
              f"(各航次: {', '.join(f'{r:.2f}' for r in rmses)})")

    bi_rmse = avg_rmse.get('双线性插值', 1.0)
    print(f"\n  M2 vs 双线性 平均改善: "
          f"{(1 - avg_rmse.get('M2 地形先验KF', bi_rmse) / bi_rmse) * 100:+.1f}%")
    ff_rmse = avg_rmse.get('频域融合', bi_rmse)
    print(f"  M2 vs 频域融合 平均改善: "
          f"{(1 - avg_rmse.get('M2 地形先验KF', ff_rmse) / ff_rmse) * 100:+.1f}%")

    # === 输出文件 ===
    save_png = OUTPUT_DIR / "m2_multi_cruise.png"
    save_txt = OUTPUT_DIR / "m2_multi_cruise_metrics.txt"

    # 分组柱状图
    fig, ax = plt.subplots(figsize=(13, 7))
    n_cruises = len(valid_cruises)
    n_methods = len(method_names)
    bar_width = 0.25
    x = np.arange(n_cruises)

    colors = {'双线性插值': '#A6BDDB', '频域融合': '#74ADD1', 'M2 地形先验KF': '#D7301F'}
    for j, name in enumerate(method_names):
        rmses = [next(m for n, m in all_results[c] if n == name)['RMSE']
                 for c in valid_cruises]
        bars = ax.bar(x + j * bar_width, rmses, bar_width,
                      label=name, color=colors.get(name, 'gray'),
                      edgecolor='none')
        for bar, v in zip(bars, rmses):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8.5,
                    fontweight='bold' if name == 'M2 地形先验KF' else 'normal')

    cruise_labels = [c.replace("_gravity_dmo", "").replace("_gravity", "")
                     for c in valid_cruises]
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(cruise_labels, fontsize=10)
    ax.set_ylabel('RMSE (mGal)', fontsize=11)
    ax.set_title(f'M2 多航次独立验证 — 20°×16° 区域 (5 条 MR/KS 系列航次)\n'
                 f'α={alpha}, 3 方法对比', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  分组柱状图: {save_png}")

    # 数值表
    with open(save_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 110 + "\n")
        f.write(f"  M2 多航次独立验证 — 20°×16° 区域\n")
        f.write(f"  区域: lon={lon_range}, lat={lat_range}\n")
        f.write(f"  α (M2 地形先验): {alpha}\n")
        f.write(f"  验证航次: {', '.join(valid_cruises)}\n")
        f.write("=" * 110 + "\n")
        for cruise in valid_cruises:
            cruise_short = cruise.replace("_gravity_dmo", "").replace("_gravity", "")
            f.write(f"\n  [{cruise_short}] (n_val={all_results[cruise][0][1]['n']:,})\n")
            f.write(f"    {'Method':<22} {'RMSE':<10} {'MAE':<10} {'Max|e|':<10} "
                    f"{'STD':<10} {'Corr':<10} {'Bias':<10} {'Time(s)':<8}\n")
            f.write("    " + "-" * 90 + "\n")
            for name, m in all_results[cruise]:
                f.write(f"    {name:<22} {m['RMSE']:<10.3f} {m['MAE']:<10.3f} "
                        f"{m['Max|e|']:<10.2f} {m['STD']:<10.3f} {m['Corr']:<10.4f} "
                        f"{m['Bias']:<+10.2f} {m['time']:<8.1f}\n")
        f.write("\n" + "=" * 110 + "\n")
        f.write(f"  平均 RMSE 汇总\n")
        f.write(f"  {'Method':<22} {'平均 RMSE':<12} {'vs 双线性':<12} {'vs 频域融合':<12}\n")
        f.write("  " + "-" * 60 + "\n")
        for name in method_names:
            impr_bi = (1 - avg_rmse[name] / bi_rmse) * 100
            impr_ff = (1 - avg_rmse[name] / ff_rmse) * 100
            f.write(f"  {name:<22} {avg_rmse[name]:<12.3f} {impr_bi:<+12.1f} {impr_ff:<+12.1f}\n")
        f.write("=" * 110 + "\n")
    print(f"  数值表: {save_txt}")

    print(f"\n  总用时: {time.time()-t_start:.0f}s")
    return all_results


if __name__ == "__main__":
    run_multi_cruise_val()
