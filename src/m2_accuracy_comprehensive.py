"""M2 综合精度对比 — 6 张关键精度指标图.

在 20°×16° 区域上构建 3 方法, 在 MR22-04 验证集上生成:

图 1: m2_accuracy_radar.png        — 综合精度雷达图 (5 维度 × 3 方法)
图 2: m2_scatter_density.png       — 散点密度图 (预测 vs 真值, hexbin)
图 3: m2_error_hist_qq.png         — 误差直方图 + QQ 图
图 4: m2_track_profile.png         — 沿 MR22-04 航迹剖面 (局部精度)
图 5: m2_accuracy_summary.png      — 三维度汇总 (单航次/4航次/大海域)
图 6: m2_improvement_breakdown.png — 改善百分比分解 (按航次 × 方法)
"""
import sys, time
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LogNorm

sys.path.insert(0, str(Path(__file__).parent))

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from independent_validation_v2 import VAL_CRUISE
from compare_m2_traditional import (
    load_region_data, build_three_methods, get_predictions_at_val, compute_metrics)

OUTPUT_DIR = Path(__file__).parent.parent / "output_real"

# 多航次 + 大海域硬编码结果 (来自 m2_multi_cruise_metrics.txt 和 m2_large_region_metrics.txt)
MULTI_CRUISE_RESULTS = {
    'MR22-04': {'双线性插值': 3.983, '频域融合': 3.936, 'M2 地形先验KF': 3.767},
    'MR22-02': {'双线性插值': 6.976, '频域融合': 6.800, 'M2 地形先验KF': 6.634},
    'MR23-02': {'双线性插值': 4.563, '频域融合': 4.523, 'M2 地形先验KF': 4.360},
    'MR21-02': {'双线性插值': 4.496, '频域融合': 4.301, 'M2 地形先验KF': 4.053},
}
LARGE_REGION = {'双线性插值': 3.59, '频域融合': 3.55, 'M2 地形先验KF': 3.46}

METHOD_COLORS = {
    '双线性插值': '#4575B4',
    '频域融合': '#74ADD1',
    'M2 地形先验KF': '#D7301F',
}
METHOD_MARKERS = {
    '双线性插值': 'o',
    '频域融合': 's',
    'M2 地形先验KF': '^',
}


def fig1_accuracy_radar(metrics_rows, save_png):
    """图 1: 综合精度雷达图 (5 维度 × 3 方法)."""
    print("  [图1] 综合精度雷达图...")
    # 5 个维度: RMSE, MAE, STD, Max|e|, |Bias| (越小越好)
    dims = ['RMSE', 'MAE', 'STD', 'Max|e|', '|Bias|']
    dim_labels = ['RMSE\n(mGal)', 'MAE\n(mGal)', 'STD\n(mGal)',
                  'Max|e|\n(mGal)', '|Bias|\n(mGal)']

    # 归一化: 每个维度上, 最大值=1, 最小值=0
    values = {}
    for name, m in metrics_rows:
        values[name] = [m['RMSE'], m['MAE'], m['STD'], m['Max|e|'], abs(m['Bias'])]

    dim_max = [max(values[n][i] for n in values) for i in range(5)]
    dim_min = [min(values[n][i] for n in values) for i in range(5)]
    norm_values = {}
    for name in values:
        norm_values[name] = [(values[name][i] - dim_min[i]) / (dim_max[i] - dim_min[i] + 1e-10)
                              for i in range(5)]

    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for name, m in metrics_rows:
        vals = norm_values[name] + norm_values[name][:1]
        ax.plot(angles, vals, color=METHOD_COLORS[name], lw=2.5,
                marker=METHOD_MARKERS[name], ms=10, label=name)
        ax.fill(angles, vals, color=METHOD_COLORS[name], alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, fontsize=11)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['', '', '', ''], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title('综合精度雷达图 (面积越小越好)\n'
                 '20°×16° 区域, MR22-04 验证 (n=46,702)',
                 fontsize=13, pad=20)
    ax.legend(fontsize=11, loc='upper right', bbox_to_anchor=(1.35, 1.1))

    # 添加数值标注
    for name, m in metrics_rows:
        print(f"    {name}: RMSE={m['RMSE']:.3f}, MAE={m['MAE']:.3f}, "
              f"STD={m['STD']:.3f}, Max|e|={m['Max|e|']:.2f}, |Bias|={abs(m['Bias']):.3f}")

    plt.tight_layout()
    plt.savefig(save_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    保存: {save_png}")


def fig2_scatter_density(models, val_points, lon_hr, lat_hr, save_png):
    """图 2: 散点密度图 (hexbin)."""
    print("  [图2] 散点密度图...")
    val_truth = val_points[:, 2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    all_vals = np.concatenate([val_truth, *[get_predictions_at_val(m, val_points, lon_hr, lat_hr)
                                            for m in models.values()]])
    all_vals = all_vals[~np.isnan(all_vals)]
    vmin, vmax = np.percentile(all_vals, [0.5, 99.5])

    for ax, (name, model) in zip(axes, models.items()):
        pred = get_predictions_at_val(model, val_points, lon_hr, lat_hr)
        valid = ~np.isnan(pred)
        x, y = val_truth[valid], pred[valid]
        hb = ax.hexbin(x, y, gridsize=80, cmap='YlOrRd',
                        extent=[vmin, vmax, vmin, vmax], mincnt=1,
                        norm=LogNorm())
        # y=x 参考线
        ax.plot([vmin, vmax], [vmin, vmax], 'k--', lw=1, alpha=0.7)
        # 回归线
        if len(x) > 10:
            coef = np.polyfit(x, y, 1)
            xx = np.array([vmin, vmax])
            ax.plot(xx, np.polyval(coef, xx), color=METHOD_COLORS[name],
                    lw=2, alpha=0.8, label=f'y={coef[0]:.3f}x+{coef[1]:.2f}')
        r, _ = pearsonr(x, y)
        rmse = float(np.sqrt(np.mean((y - x) ** 2)))
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_xlabel('船载真值 (mGal)', fontsize=11)
        ax.set_ylabel('模型预测 (mGal)', fontsize=11)
        ax.set_title(f'{name}\nR={r:.4f}, RMSE={rmse:.3f}', fontsize=12)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        cb = fig.colorbar(hb, ax=ax, orientation='vertical',
                          pad=0.02, shrink=0.85)
        cb.set_label('点数密度', fontsize=9)

    fig.suptitle('散点密度图 — 模型预测 vs 船载真值 (MR22-04, n=46,702)\n'
                 '虚线=y=x (理想), 实线=回归线', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    保存: {save_png}")


def fig3_error_hist_qq(models, val_points, lon_hr, lat_hr, save_png):
    """图 3: 误差直方图 + QQ 图."""
    print("  [图3] 误差直方图 + QQ 图...")
    val_truth = val_points[:, 2]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 左: 误差直方图
    ax = axes[0]
    errors = {}
    for name, model in models.items():
        pred = get_predictions_at_val(model, val_points, lon_hr, lat_hr)
        valid = ~np.isnan(pred)
        errors[name] = pred[valid] - val_truth[valid]

    bin_range = (-15, 15)
    bins = np.linspace(bin_range[0], bin_range[1], 60)
    for name, err in errors.items():
        ax.hist(err, bins=bins, alpha=0.55, label=f'{name} (σ={err.std():.2f})',
                color=METHOD_COLORS[name], density=True, edgecolor='none')

    ax.axvline(0, color='black', ls='--', lw=1, alpha=0.5)
    ax.set_xlim(bin_range)
    ax.set_xlabel('误差 (mGal)', fontsize=12)
    ax.set_ylabel('概率密度', fontsize=12)
    ax.set_title('误差直方图 (正态化)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 右: QQ 图 (vs 正态分布)
    ax = axes[1]
    from scipy.stats import norm, probplot
    for name, err in errors.items():
        # 标准化
        err_norm = (err - err.mean()) / err.std()
        # 计算 QQ 数据
        n = len(err_norm)
        quantiles = np.linspace(0.01, 0.99, 100)
        theoretical_q = norm.ppf(quantiles)
        sample_q = np.quantile(err_norm, quantiles)
        ax.scatter(theoretical_q, sample_q, s=15,
                   color=METHOD_COLORS[name], alpha=0.7, label=name)

    ax.plot([-3, 3], [-3, 3], 'k--', lw=1, alpha=0.7, label='正态分布参考')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('理论正态分位数', fontsize=12)
    ax.set_ylabel('样本分位数 (标准化后)', fontsize=12)
    ax.set_title('QQ 图 (vs 正态分布)\n偏离直线 = 误差非正态', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    fig.suptitle('误差分布对比 — 直方图 + QQ 图 (MR22-04)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    保存: {save_png}")


def fig4_track_profile(models, val_points, lon_hr, lat_hr, save_png):
    """图 4: 沿 MR22-04 航迹剖面图."""
    print("  [图4] 沿航迹剖面图...")
    # 按时间顺序排 (val_points 已是航次顺序, 取连续 2000 点)
    n = len(val_points)
    start, end = n // 4, n // 4 + 2000  # 取中间段
    pts = val_points[start:end]

    val_truth = pts[:, 2]
    # 沿航迹距离 (km)
    dlon = np.diff(pts[:, 0])
    dlat = np.diff(pts[:, 1])
    cos_lat = np.cos(np.radians(pts[:-1, 1]))
    distance = np.cumsum(np.sqrt((dlon * cos_lat) ** 2 + dlat ** 2) * 111.32)
    distance = np.concatenate([[0], distance])

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                              gridspec_kw={'height_ratios': [2, 1]})

    # 上: 重力剖面
    ax = axes[0]
    ax.plot(distance, val_truth, 'k-', lw=1.5, alpha=0.7, label='船载真值')
    for name, model in models.items():
        pred = get_predictions_at_val(model, pts, lon_hr, lat_hr)
        ax.plot(distance, pred, '-', lw=1.2, alpha=0.85,
                color=METHOD_COLORS[name], label=name)
    ax.set_ylabel('FAA (mGal)', fontsize=12)
    ax.set_title(f'沿 MR22-04 航迹重力剖面 (n={len(pts)}, 起点索引={start})',
                 fontsize=13)
    ax.legend(fontsize=10, loc='upper right', ncol=4)
    ax.grid(True, alpha=0.3)

    # 下: 误差剖面
    ax = axes[1]
    for name, model in models.items():
        pred = get_predictions_at_val(model, pts, lon_hr, lat_hr)
        err = pred - val_truth
        ax.plot(distance, err, '-', lw=1, alpha=0.85,
                color=METHOD_COLORS[name], label=name)
    ax.axhline(0, color='black', ls='--', lw=1, alpha=0.5)
    ax.fill_between(distance, -3, 3, color='gray', alpha=0.15, label='±3 mGal 带')
    ax.set_xlabel('沿航迹距离 (km)', fontsize=12)
    ax.set_ylabel('误差 (mGal)', fontsize=12)
    ax.set_title('误差剖面', fontsize=12)
    ax.legend(fontsize=10, loc='upper right', ncol=4)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-15, 15)

    plt.tight_layout()
    plt.savefig(save_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    保存: {save_png}")


def fig5_accuracy_summary(save_png):
    """图 5: 综合精度汇总图 (3 维度 × 3 方法)."""
    print("  [图5] 综合精度汇总图...")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    method_names = ['双线性插值', '频域融合', 'M2 地形先验KF']

    # 左: 3 个维度的 RMSE 对比
    ax = axes[0]
    dimensions = ['20°×16°\n(MR22-04)', '4 航次\n平均', '30°×24°\n大海域']
    rmse_data = {m: [] for m in method_names}
    for m in method_names:
        rmse_data[m].append(MULTI_CRUISE_RESULTS['MR22-04'][m])
        rmse_data[m].append(np.mean([MULTI_CRUISE_RESULTS[c][m]
                                       for c in MULTI_CRUISE_RESULTS]))
        rmse_data[m].append(LARGE_REGION[m])

    x = np.arange(3)
    width = 0.25
    for i, m in enumerate(method_names):
        bars = ax.bar(x + (i - 1) * width, rmse_data[m], width,
                      label=m, color=METHOD_COLORS[m], edgecolor='none')
        for bar, v in zip(bars, rmse_data[m]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=10,
                    fontweight='bold' if m == 'M2 地形先验KF' else 'normal')
    ax.set_xticks(x)
    ax.set_xticklabels(dimensions, fontsize=11)
    ax.set_ylabel('RMSE (mGal)', fontsize=12)
    ax.set_title('RMSE 汇总 — 3 维度 × 3 方法', fontsize=13)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # 右: 改善百分比
    ax = axes[1]
    improvements = {m: [] for m in method_names}
    for m in method_names:
        # vs 双线性
        improvements[m].append((1 - rmse_data[m][0] / rmse_data['双线性插值'][0]) * 100)
        improvements[m].append((1 - rmse_data[m][1] / rmse_data['双线性插值'][1]) * 100)
        improvements[m].append((1 - rmse_data[m][2] / rmse_data['双线性插值'][2]) * 100)

    for i, m in enumerate(method_names):
        if m == '双线性插值':
            continue
        bars = ax.bar(x + (i - 1) * width, improvements[m], width,
                      label=f'{m} vs 双线性', color=METHOD_COLORS[m],
                      edgecolor='none')
        for bar, v in zip(bars, improvements[m]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{v:+.1f}%', ha='center', va='bottom', fontsize=10,
                    fontweight='bold' if m == 'M2 地形先验KF' else 'normal')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(dimensions, fontsize=11)
    ax.set_ylabel('改善百分比 (%)', fontsize=12)
    ax.set_title('改善百分比 — M2 vs 双线性 (3 维度)', fontsize=13)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    fig.suptitle('M2 综合精度汇总 — 3 验证维度 × 3 方法',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    保存: {save_png}")


def fig6_improvement_breakdown(save_png):
    """图 6: 改善百分比分解 (按航次 × 方法)."""
    print("  [图6] 改善百分比分解...")
    cruises = list(MULTI_CRUISE_RESULTS.keys())
    method_names = ['频域融合', 'M2 地形先验KF']

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 左: 各航次 M2 vs 双线性 改善
    ax = axes[0]
    impr_vs_bi = [(1 - MULTI_CRUISE_RESULTS[c]['M2 地形先验KF'] /
                    MULTI_CRUISE_RESULTS[c]['双线性插值']) * 100 for c in cruises]
    impr_vs_ff = [(1 - MULTI_CRUISE_RESULTS[c]['M2 地形先验KF'] /
                    MULTI_CRUISE_RESULTS[c]['频域融合']) * 100 for c in cruises]
    x = np.arange(len(cruises))
    width = 0.35
    bars1 = ax.bar(x - width/2, impr_vs_bi, width, label='M2 vs 双线性',
                   color='#D7301F', edgecolor='none')
    bars2 = ax.bar(x + width/2, impr_vs_ff, width, label='M2 vs 频域融合',
                   color='#FC8D59', edgecolor='none')
    for bar, v in zip(bars1, impr_vs_bi):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{v:+.1f}%', ha='center', va='bottom', fontsize=10,
                fontweight='bold')
    for bar, v in zip(bars2, impr_vs_ff):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{v:+.1f}%', ha='center', va='bottom', fontsize=10,
                fontweight='bold')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cruises, fontsize=10)
    ax.set_ylabel('改善百分比 (%)', fontsize=12)
    ax.set_title('M2 改善 — 按航次分解', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # 右: 各航次 3 方法 RMSE 折线
    ax = axes[1]
    for m in ['双线性插值', '频域融合', 'M2 地形先验KF']:
        rmses = [MULTI_CRUISE_RESULTS[c][m] for c in cruises]
        ax.plot(cruises, rmses, 'o-', lw=2, ms=10,
                color=METHOD_COLORS[m], label=m)
    ax.set_xlabel('验证航次', fontsize=12)
    ax.set_ylabel('RMSE (mGal)', fontsize=12)
    ax.set_title('3 方法 RMSE — 按航次对比', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    fig.suptitle('M2 改善分解 — 4 条独立航次 (α=0.2)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    保存: {save_png}")


def main(lon_range=(132, 152), lat_range=(26, 42), upscale=4, alpha=0.2):
    print("=" * 70)
    print("  M2 综合精度对比 — 6 张精度指标图")
    print("=" * 70)
    t_start = time.time()

    print(f"\n[1/3] 加载数据 + 构建 3 方法 (α={alpha})...")
    data = load_region_data(lon_range, lat_range, val_cruise=VAL_CRUISE, upscale=upscale)
    data['lon_range'] = lon_range
    data['lat_range'] = lat_range
    models, _ = build_three_methods(data, alpha=alpha)

    val_points = data['val_points']
    lon_hr, lat_hr = data['lon_hr'], data['lat_hr']
    val_truth = val_points[:, 2]

    # 计算各方法 metrics
    metrics_rows = []
    for name in models:
        pred = get_predictions_at_val(models[name], val_points, lon_hr, lat_hr)
        m = compute_metrics(pred, val_truth)
        metrics_rows.append((name, m))

    print(f"\n[2/3] 各方法精度 (MR22-04, n={len(val_points):,}):")
    print(f"  {'Method':<22} {'RMSE':<10} {'MAE':<10} {'STD':<10} "
          f"{'Max|e|':<10} {'Corr':<10} {'Bias':<10}")
    for name, m in metrics_rows:
        print(f"  {name:<22} {m['RMSE']:<10.3f} {m['MAE']:<10.3f} "
              f"{m['STD']:<10.3f} {m['Max|e|']:<10.2f} {m['Corr']:<10.4f} "
              f"{m['Bias']:<+10.2f}")

    print(f"\n[3/3] 生成 6 张图...")
    fig1_accuracy_radar(metrics_rows, OUTPUT_DIR / "m2_accuracy_radar.png")
    fig2_scatter_density(models, val_points, lon_hr, lat_hr,
                         OUTPUT_DIR / "m2_scatter_density.png")
    fig3_error_hist_qq(models, val_points, lon_hr, lat_hr,
                       OUTPUT_DIR / "m2_error_hist_qq.png")
    fig4_track_profile(models, val_points, lon_hr, lat_hr,
                       OUTPUT_DIR / "m2_track_profile.png")
    fig5_accuracy_summary(OUTPUT_DIR / "m2_accuracy_summary.png")
    fig6_improvement_breakdown(OUTPUT_DIR / "m2_improvement_breakdown.png")

    print(f"\n  总用时: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.2)
    args = parser.parse_args()
    main(alpha=args.alpha)
