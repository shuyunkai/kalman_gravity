"""多航次独立验证 — 3 方法 (双线性 / 频域融合 / KF) 精度对比.

每条航次轮流做验证集, 其余作训练, 在 20°×16° 区域上跑 3 方法:
  - MR22-04_gravity_dmo  (日本海沟, 主验证)
  - MR22-02_gravity_dmo
  - MR23-02_gravity_dmo
  - MR21-02_gravity_dmo

输出:
  1. fig_per_cruise_metrics.png — 各航次上 3 方法多维度精度对比 (4 子图 + 总平均)
  2. fig_summary_metrics.png    — 总航线汇总 3 方法精度对比 (5 维度柱状图 + 指标表)
  3. validation_metrics.txt     — 完整数值表
"""
import sys, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).parent))

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from three_method_compare import (
    load_region_data, build_three_methods, get_predictions_at_val,
    compute_metrics, METHOD_BILINEAR, METHOD_FREQFUSION, METHOD_KF)

OUTPUT_DIR = Path(__file__).parent.parent / "output_real"

VAL_CRUISES = [
    "MR22-04_gravity_dmo",
    "MR22-02_gravity_dmo",
    "MR23-02_gravity_dmo",
    "MR21-02_gravity_dmo",
]

METRICS_TO_PLOT = ['RMSE', 'MAE', 'STD', 'Max|e|']
METHODS = [METHOD_BILINEAR, METHOD_FREQFUSION, METHOD_KF]
COLORS = {METHOD_BILINEAR: '#A6BDDB', METHOD_FREQFUSION: '#74ADD1', METHOD_KF: '#D7301F'}


def run_multi_cruise_val(lon_range=(132, 152), lat_range=(26, 42),
                         upscale=4, alpha=0.5):
    print("=" * 70)
    print("  多航次独立验证 — 3 方法精度对比")
    print("=" * 70)
    t_start = time.time()

    all_results = {}  # {cruise: [(method, metrics), ...]}
    # 累积逐点预测/真值: 总汇总必须把四条航次的点直接拼起来重算,
    # 不能用"对各航次指标再做一次平均"的公式近似(见下方注释)。
    pooled = {name: {'pred': [], 'truth': []} for name in METHODS}

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

        models, times = build_three_methods(data, alpha=alpha)

        val_truth = data['val_points'][:, 2]
        cruise_results = []
        for name in METHODS:
            pred = get_predictions_at_val(models[name], data['val_points'],
                                          data['lon_hr'], data['lat_hr'])
            m = compute_metrics(pred, val_truth)
            m['time'] = times[name]
            cruise_results.append((name, m))
            pooled[name]['pred'].append(pred)
            pooled[name]['truth'].append(val_truth)
            print(f"  {name:<12} RMSE={m['RMSE']:.3f}  MAE={m['MAE']:.3f}  "
                  f"Corr={m['Corr']:.4f}")

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

    # 各航次 × 各方法 RMSE
    print(f"\n  {'航次':<15}", end="")
    for n in METHODS:
        print(f"  {n:<12}", end="")
    print()
    print("  " + "-" * 60)
    for cruise in valid_cruises:
        cruise_short = cruise.replace("_gravity_dmo", "").replace("_gravity", "")
        print(f"  {cruise_short:<15}", end="")
        for name, m in all_results[cruise]:
            print(f"  RMSE={m['RMSE']:.3f}", end="  ")
        print()

    # 各方法总汇总（逐点合并，而不是对各航次指标再做平均）
    #
    # 原实现把 RMSE/MAE/STD/Max|e| 一律按 sqrt(sum(n*v^2)/sum(n)) 聚合，只有 RMSE 恰好成立：
    #   * MAE 是 |e| 的均值, 合并应为 n 加权算术平均 (原式系统性偏大)
    #   * STD 合并需含航次间偏差: var = sum(n*(std^2+bias^2))/N - bias_all^2
    #   * Max|e| 是极值, 不可"均方平均" —— 原式给出的 26.22 mGal 比它自己
    #     三条子样本的最大值(28.42/33.80/21.65)还小, 不是任何东西的最大值
    #   * Corr 也无法由分组相关系数还原
    # 正确做法: 把四条航次的逐点预测/真值直接拼接后重算指标。
    print(f"\n  各方法总汇总 (逐点合并):")
    avg_metrics = {}
    for name in METHODS:
        pred_all = np.concatenate(pooled[name]['pred'])
        truth_all = np.concatenate(pooled[name]['truth'])
        m = compute_metrics(pred_all, truth_all)
        m['time'] = float(np.sum([next(mm for nn, mm in all_results[c] if nn == name)['time']
                                  for c in valid_cruises]))
        avg_metrics[name] = m
        print(f"    {name:<12}: RMSE={m['RMSE']:.3f}  "
              f"MAE={m['MAE']:.3f}  "
              f"STD={m['STD']:.3f}  "
              f"Max|e|={m['Max|e|']:.2f}  "
              f"Corr={m['Corr']:.4f}  "
              f"Bias={m['Bias']:+.3f}  "
              f"(n={m['n']:,})")

    bi_rmse = avg_metrics[METHOD_BILINEAR]['RMSE']
    ff_rmse = avg_metrics[METHOD_FREQFUSION]['RMSE']
    print(f"\n  KF vs 双线性:  RMSE 改善 {(1-avg_metrics[METHOD_KF]['RMSE']/bi_rmse)*100:+.1f}%")
    print(f"  KF vs 频域融合: RMSE 改善 {(1-avg_metrics[METHOD_KF]['RMSE']/ff_rmse)*100:+.1f}%")

    # === 生成图 ===
    _plot_per_cruise(all_results, valid_cruises, alpha, lon_range, lat_range)
    _plot_summary(all_results, valid_cruises, avg_metrics, alpha, lon_range, lat_range)
    _save_metrics_txt(all_results, valid_cruises, avg_metrics, alpha,
                      lon_range, lat_range)

    print(f"\n  总用时: {time.time()-t_start:.0f}s")
    return all_results


def _plot_per_cruise(all_results, valid_cruises, alpha, lon_range, lat_range):
    """图 1: 各航次上 3 方法多维度精度对比."""
    print("\n  生成图: 各航次精度对比...")
    n_cruises = len(valid_cruises)
    n_metrics = len(METRICS_TO_PLOT)

    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

    cruise_labels = [c.replace("_gravity_dmo", "").replace("_gravity", "")
                     for c in valid_cruises]

    for idx, metric in enumerate(METRICS_TO_PLOT):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        x = np.arange(n_cruises)
        width = 0.25
        for i, method in enumerate(METHODS):
            vals = [next(m for n, m in all_results[c] if n == method)[metric]
                    for c in valid_cruises]
            bars = ax.bar(x + (i - 1) * width, vals, width,
                          label=method, color=COLORS[method], edgecolor='none')
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + max(vals) * 0.015,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=8.5,
                        fontweight='bold' if method == METHOD_KF else 'normal')
        ax.set_xticks(x)
        ax.set_xticklabels(cruise_labels, fontsize=10)
        ax.set_ylabel(f'{metric} (mGal)' if metric != 'Corr' else metric, fontsize=11)
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    fig.suptitle(f'各航次上 3 方法精度对比 — 20°×16° 区域 (α={alpha})\n'
                 f'验证航次: {", ".join(cruise_labels)} (完全独立未参与训练)',
                 fontsize=13, fontweight='bold', y=0.995)
    save_png = OUTPUT_DIR / "fig_per_cruise_metrics.png"
    plt.savefig(save_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    保存: {save_png}")


def _plot_summary(all_results, valid_cruises, avg_metrics, alpha, lon_range, lat_range):
    """图 2: 总航线汇总 3 方法精度对比 (5 维度柱状图 + 指标表)."""
    print("  生成图: 总汇总精度对比...")
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.25,
                  width_ratios=[1.3, 1.0])

    # 左上: 4 维度柱状图 (汇总)
    ax1 = fig.add_subplot(gs[0, 0])
    metrics_4d = ['RMSE', 'MAE', 'STD', 'Max|e|']
    x = np.arange(len(metrics_4d))
    width = 0.25
    for i, method in enumerate(METHODS):
        vals = [avg_metrics[method]['RMSE'], avg_metrics[method]['MAE'],
                avg_metrics[method]['STD'], avg_metrics[method]['Max|e|']]
        bars = ax1.bar(x + (i - 1) * width, vals, width,
                       label=method, color=COLORS[method], edgecolor='none')
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + max(vals) * 0.02,
                     f'{v:.2f}', ha='center', va='bottom', fontsize=9,
                     fontweight='bold' if method == METHOD_KF else 'normal')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_4d, fontsize=10)
    ax1.set_ylabel('汇总值 (mGal)', fontsize=11)
    ax1.set_title(f'(a) 总汇总 4 维度精度对比 (n={avg_metrics[METHOD_KF]["n"]:,} 点)',
                  fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)

    # 右上: 各航次 RMSE 折线 + 总平均点
    ax2 = fig.add_subplot(gs[0, 1])
    cruise_labels = [c.replace("_gravity_dmo", "").replace("_gravity", "")
                     for c in valid_cruises]
    x = np.arange(len(cruise_labels) + 1)  # +1 for 平均
    for method in METHODS:
        vals = [next(m for n, m in all_results[c] if n == method)['RMSE']
                for c in valid_cruises]
        vals.append(avg_metrics[method]['RMSE'])
        ax2.plot(x, vals, 'o-', color=COLORS[method], label=method,
                 lw=2, markersize=9,
                 markeredgecolor='white', markeredgewidth=1.2)
        for xi, v in zip(x, vals):
            ax2.text(xi, v + 0.08, f'{v:.2f}', ha='center', va='bottom',
                     fontsize=8.5,
                     fontweight='bold' if method == METHOD_KF else 'normal')
    ax2.set_xticks(x)
    ax2.set_xticklabels(cruise_labels + ['总平均'], fontsize=10)
    ax2.set_ylabel('RMSE (mGal)', fontsize=11)
    ax2.set_title('(b) 各航次 RMSE + 总平均', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    for spine in ['top', 'right']:
        ax2.spines[spine].set_visible(False)

    # 左下: 改善百分比
    ax3 = fig.add_subplot(gs[1, 0])
    metrics_impr = ['RMSE', 'MAE', 'STD', 'Max|e|']
    x = np.arange(len(metrics_impr))
    width = 0.35
    impr_vs_bi = [(1 - avg_metrics[METHOD_KF][m] / avg_metrics[METHOD_BILINEAR][m]) * 100
                  for m in metrics_impr]
    impr_vs_ff = [(1 - avg_metrics[METHOD_KF][m] / avg_metrics[METHOD_FREQFUSION][m]) * 100
                  for m in metrics_impr]
    bars1 = ax3.bar(x - width/2, impr_vs_bi, width,
                    label='KF vs 双线性', color='#D7301F', edgecolor='none')
    bars2 = ax3.bar(x + width/2, impr_vs_ff, width,
                    label='KF vs 频域融合', color='#FC8D59', edgecolor='none')
    for bar, v in zip(bars1, impr_vs_bi):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                 f'{v:+.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, v in zip(bars2, impr_vs_ff):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                 f'{v:+.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics_impr, fontsize=10)
    ax3.set_ylabel('改善百分比 (%)', fontsize=11)
    ax3.set_title('(c) KF 改善百分比 (正值=KF 更优)', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9, loc='upper right')
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(0, color='gray', lw=0.8)
    for spine in ['top', 'right']:
        ax3.spines[spine].set_visible(False)

    # 右下: 指标表
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    cell_text = []
    for metric in ['RMSE', 'MAE', 'STD', 'Max|e|', 'Corr', 'n']:
        row = [metric]
        for method in METHODS:
            v = avg_metrics[method][metric]
            if metric == 'n':
                row.append(f'{v:,}')
            elif metric == 'Corr':
                row.append(f'{v:.4f}')
            else:
                row.append(f'{v:.3f}')
        cell_text.append(row)
    col_labels = ['指标'] + METHODS
    table = ax4.table(cellText=cell_text, colLabels=col_labels,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 1.7)

    # KF 列高亮 + 最优值标红
    n_rows = len(cell_text) + 1
    for i in range(1, n_rows):
        cell = table[i, 3]
        cell.set_facecolor('#FFEDA0')
        cell.set_text_props(fontweight='bold')
        # 最优 (越小越好; Corr 越大越好)
        metric_name = cell_text[i-1][0]
        if metric_name in ['RMSE', 'MAE', 'STD', 'Max|e|']:
            vals = [float(cell_text[i-1][j]) for j in range(1, 4)]
            best_idx = vals.index(min(vals)) + 1
            if best_idx != 3:
                table[i, best_idx].set_text_props(fontweight='bold', color='#D7301F')
        elif metric_name == 'Corr':
            vals = [float(cell_text[i-1][j]) for j in range(1, 4)]
            best_idx = vals.index(max(vals)) + 1
            if best_idx != 3:
                table[i, best_idx].set_text_props(fontweight='bold', color='#D7301F')

    for j in range(4):
        cell = table[0, j]
        cell.set_facecolor('#3690C0')
        cell.set_text_props(color='white', fontweight='bold')

    ax4.set_title('(d) 总汇总指标表 (黄=KF, 红=该维度最优)',
                  fontsize=11, fontweight='bold', pad=12)

    fig.suptitle(f'3 方法精度对比 — 总汇总 (4 条独立航次, α={alpha})\n'
                 f'区域: lon={lon_range}, lat={lat_range}',
                 fontsize=13, fontweight='bold', y=0.995)
    save_png = OUTPUT_DIR / "fig_summary_metrics.png"
    plt.savefig(save_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    保存: {save_png}")


def _save_metrics_txt(all_results, valid_cruises, avg_metrics, alpha,
                      lon_range, lat_range):
    """完整数值表."""
    save_txt = OUTPUT_DIR / "validation_metrics.txt"
    with open(save_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 95 + "\n")
        f.write(f"  3 方法多航次独立验证\n")
        f.write(f"  区域: lon={lon_range}, lat={lat_range} (20°×16°)\n")
        f.write(f"  α (KF 地形先验权重): {alpha}\n")
        f.write(f"  验证航次: {', '.join(valid_cruises)}\n")
        f.write("=" * 95 + "\n")
        for cruise in valid_cruises:
            cruise_short = cruise.replace("_gravity_dmo", "").replace("_gravity", "")
            f.write(f"\n  [{cruise_short}] (n_val={all_results[cruise][0][1]['n']:,})\n")
            f.write(f"    {'Method':<14} {'RMSE':<10} {'MAE':<10} {'Max|e|':<10} "
                    f"{'STD':<10} {'Corr':<10} {'Bias':<10} {'Time(s)':<8}\n")
            f.write("    " + "-" * 85 + "\n")
            for name, m in all_results[cruise]:
                f.write(f"    {name:<14} {m['RMSE']:<10.3f} {m['MAE']:<10.3f} "
                        f"{m['Max|e|']:<10.2f} {m['STD']:<10.3f} {m['Corr']:<10.4f} "
                        f"{m['Bias']:<+10.3f} {m['time']:<8.1f}\n")
        f.write("\n" + "=" * 95 + "\n")
        f.write(f"  总汇总 (4 航次逐点合并, n={avg_metrics[METHOD_KF]['n']:,} 点)\n")
        f.write(f"    {'Method':<14} {'RMSE':<10} {'MAE':<10} {'Max|e|':<10} "
                f"{'STD':<10} {'Corr':<10} {'Bias':<10}\n")
        f.write("    " + "-" * 75 + "\n")
        for method in METHODS:
            m = avg_metrics[method]
            f.write(f"    {method:<14} {m['RMSE']:<10.3f} {m['MAE']:<10.3f} "
                    f"{m['Max|e|']:<10.2f} {m['STD']:<10.3f} {m['Corr']:<10.4f} "
                    f"{m['Bias']:<+10.3f}\n")
        f.write("    注: RMSE_bc (去偏差 RMSE) 恒等于 STD; Bias 为系统偏差(基准不一致)\n")
        f.write("\n")
        bi_rmse = avg_metrics[METHOD_BILINEAR]['RMSE']
        ff_rmse = avg_metrics[METHOD_FREQFUSION]['RMSE']
        kf_rmse = avg_metrics[METHOD_KF]['RMSE']
        f.write(f"  KF vs 双线性:   RMSE 改善 {(1-kf_rmse/bi_rmse)*100:+.1f}%  "
                f"({bi_rmse:.3f} → {kf_rmse:.3f})\n")
        f.write(f"  KF vs 频域融合: RMSE 改善 {(1-kf_rmse/ff_rmse)*100:+.1f}%  "
                f"({ff_rmse:.3f} → {kf_rmse:.3f})\n")
        f.write("=" * 95 + "\n")
    print(f"  数值表: {save_txt}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="KF 地形先验权重 (默认 0.2)")
    args = parser.parse_args()
    run_multi_cruise_val(alpha=args.alpha)
