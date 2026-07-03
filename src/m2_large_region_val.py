"""M2 大海域验证 — 30°×24° 日本周边扩展.

区域: lon=128-158°E, lat=24-48°N (30°×24°, 在 20°×16° 基础上向外扩展)
- 1' 网格: 1800×1440
- 15" 网格: 7200×5760 = 41M 像素
- 验证: MR22-04 (主验证航次, 与对比实验一致)
- 内存预算: KF 状态 ~1GB, update_shipborne correction_field ~1.5GB

跑 3 方法: 双线性 / 频域融合 / M2 (用阶段 3 最优 α)

若 30°×24° 内存溢出, 自动回退到 26°×22° (lon=130-156, lat=24-46).

输出:
  - output_real/m2_large_region.png
  - output_real/m2_large_region_metrics.txt
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
    load_region_data, build_three_methods, get_predictions_at_val,
    compute_metrics, plot_comparison)

OUTPUT_DIR = Path(__file__).parent.parent / "output_real"


def main(lon_range=(128, 158), lat_range=(24, 48), upscale=4, alpha=0.5,
         val_cruise=VAL_CRUISE):
    print("=" * 70)
    print(f"  M2 大海域验证 — 30°×24° 日本周边扩展")
    print("=" * 70)
    t_start = time.time()

    print(f"\n[1/3] 加载数据 (区域 lon={lon_range}, lat={lat_range})...")
    try:
        data = load_region_data(lon_range, lat_range, val_cruise=val_cruise,
                                upscale=upscale)
        data['lon_range'] = lon_range
        data['lat_range'] = lat_range
    except MemoryError as e:
        print(f"  30°×24° 内存溢出, 回退到 26°×22° (lon=130-156, lat=24-46)")
        lon_range = (130, 156)
        lat_range = (24, 46)
        data = load_region_data(lon_range, lat_range, val_cruise=val_cruise,
                                upscale=upscale)
        data['lon_range'] = lon_range
        data['lat_range'] = lat_range

    print(f"\n  区域大小: {lat_range[1]-lat_range[0]}°×{lon_range[1]-lon_range[0]}°")
    print(f"  15'' 网格: {data['ny_hr']}×{data['nx_hr']} = "
          f"{data['ny_hr']*data['nx_hr']/1e6:.1f}M 像素")

    print(f"\n[2/3] 构建 3 方法...")
    models, times = build_three_methods(data, alpha=alpha, n_iter=6,
                                         smooth_sigma=15.0)

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

    save_png = OUTPUT_DIR / "m2_large_region.png"
    save_txt = OUTPUT_DIR / "m2_large_region_metrics.txt"
    plot_comparison(models, times, metrics_rows, data, save_png, save_txt,
                    lon_range, lat_range, val_cruise)

    print(f"\n  总用时: {time.time()-t_start:.0f}s")
    return models, metrics_rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    main(alpha=args.alpha)
