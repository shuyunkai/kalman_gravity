"""M2 vs 传统方法 — 大区域精度对比.

区域: lon=132-152°E, lat=26-42°N (20°×16°, 较前次 14°×14° 扩大 ~80%)
- 1' 网格: 1200×960
- 15" 网格: 4800×3840 = 18.4M 像素
- 验证: MR22-04 独立巡航 (完全未参与训练)

对比方法 (5 个):
  传统方法:
    1. 双线性插值 (SSV 1' 上采样)
    2. 频域融合 (Butterworth 滤波器, SSV 低频 + 船载高频)
    3. 克里金 + 趋势面 (普通克里金, 球面变差函数)
  KF 方法:
    4. 原始 KF (KalmanGravityReconstructor, SSV 双线性先验)
    5. M2 地形先验 KF (BathyPriorKF α=0.5, 本次新方法)

所有方法在同一 MR22-04 验证集上评估, 保证公平.
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
from run_pipeline import load_ssv_region, load_gebco_region
from freq_fusion_baseline import butterworth_2d_filter
from kalman_gravity import KalmanGravityReconstructor
from kf_novel import BathyPriorKF, PureKF
from kriging_baseline import _subsample_train_points, _residual_to_faa

OUTPUT_DIR = Path(__file__).parent.parent / "output_real"


def get_predictions_at_val(gravity_15s, val_points, lon_hr, lat_hr):
    interp = RegularGridInterpolator(
        (lat_hr, lon_hr), gravity_15s, bounds_error=False, fill_value=np.nan)
    return interp(np.column_stack([val_points[:, 1], val_points[:, 0]]))


def load_bathy_15s(lon_range, lat_range, lon_hr, lat_hr):
    bathy_lr, lon_b, lat_b = load_gebco_region(lon_range, lat_range)
    interp = RegularGridInterpolator(
        (lat_b, lon_b), bathy_lr, bounds_error=False, fill_value=0)
    lon_mesh, lat_mesh = np.meshgrid(lon_hr, lat_hr)
    return interp(np.column_stack([lat_mesh.ravel(), lon_mesh.ravel()])).reshape(
        len(lat_hr), len(lon_hr))


def compute_metrics(pred, truth):
    valid = ~(np.isnan(pred) | np.isnan(truth))
    diff = pred[valid] - truth[valid]
    return {
        'RMSE': float(np.sqrt(np.mean(diff ** 2))),
        'MAE': float(np.mean(np.abs(diff))),
        'Max|e|': float(np.max(np.abs(diff))),
        'STD': float(np.std(diff)),
        'Corr': float(np.corrcoef(pred[valid], truth[valid])[0, 1]),
        'Bias': float(np.mean(diff)),
        'n': int(valid.sum()),
    }


def kriging_with_external_train(gravity_lr, ship_grav, train_mask,
                                  lon_hr, lat_hr, lon_range, lat_range, upscale):
    """克里金 + 趋势面, 使用外部 train_mask (排除 MR22-04).

    优化: 大区域内存控制 — 在 2' 网格上做克里金, 再上采样到 15".
    趋势面: 10 项三次多项式 (大区域需要更高阶).
    """
    from pykrige.ok import OrdinaryKriging
    ny_hr, nx_hr = ship_grav.shape
    ny_lr, nx_lr = gravity_lr.shape

    # 子采样训练点 (大区域 4000, 平衡精度与内存)
    n_target = 4000 if ny_lr * nx_lr > 500_000 else 6000
    lon_pts, lat_pts, val_pts = _subsample_train_points(
        lon_hr, lat_hr, ship_grav, train_mask, n_target=n_target, seed=42)
    print(f"    Kriging train points (subsampled): {len(lon_pts):,}")

    # 趋势面 (10 项三次多项式, 标准化特征以避免数值不稳定)
    lat0, lon0 = lat_range[0], lon_range[0]
    dlat = lat_pts - lat0
    dlon = lon_pts - lon0
    A_train = np.column_stack([
        np.ones_like(dlat),
        dlat, dlon,
        dlat * dlon, dlat**2, dlon**2,
        dlat**2 * dlon, dlat * dlon**2,
        dlat**3, dlon**3
    ])
    # 标准化非常数列 (大区域多项式特征尺度差异大)
    col_means = A_train[:, 1:].mean(axis=0)
    col_stds = A_train[:, 1:].std(axis=0) + 1e-10
    A_train_norm = A_train.copy()
    A_train_norm[:, 1:] = (A_train[:, 1:] - col_means) / col_stds
    coef_norm, *_ = np.linalg.lstsq(A_train_norm, val_pts, rcond=None)
    resid_train = val_pts - A_train_norm @ coef_norm
    resid_var = np.var(resid_train)
    nugget = 5.0
    sill = max(resid_var, nugget + 1.0)
    rng_a = 0.25  # 25 km range
    print(f"    Variogram: nugget={nugget}, range={rng_a}°, sill={sill:.1f}, "
          f"resid std={np.std(resid_train):.2f} mGal")

    # 克里金插值残差 — 在 2' 网格上 (大区域内存控制)
    kr_factor = 2  # 2' 网格
    ny_kr = ny_lr // kr_factor
    nx_kr = nx_lr // kr_factor
    lon_kr = np.linspace(lon_range[0], lon_range[1], nx_kr)
    lat_kr = np.linspace(lat_range[0], lat_range[1], ny_kr)
    print(f"    Kriging grid: {ny_kr}×{nx_kr} = {ny_kr*nx_kr:,} points "
          f"({kr_factor}' resolution)")
    print(f"    Memory estimate: {ny_kr*nx_kr*len(lon_pts)*8/1024**3:.2f} GiB")

    OK = OrdinaryKriging(
        lon_pts, lat_pts, resid_train,
        variogram_model='spherical',
        variogram_parameters={'nugget': nugget, 'range': rng_a, 'sill': sill},
        verbose=False, enable_plotting=False,
        exact_values=False, pseudo_inv=True)
    z_krig_2m, _ = OK.execute('grid', lon_kr, lat_kr, backend='vectorized')
    z_krig_2m = np.asarray(z_krig_2m)
    if z_krig_2m.shape != (ny_kr, nx_kr):
        z_krig_2m = z_krig_2m.T

    # 趋势面 (2' 网格, 标准化后还原)
    ii_2m, jj_2m = np.meshgrid(np.arange(ny_kr), np.arange(nx_kr), indexing='ij')
    dlat_g2 = lat_kr[ii_2m] - lat0
    dlon_g2 = lon_kr[jj_2m] - lon0
    A_grid_2m = np.column_stack([
        np.ones(ny_kr * nx_kr),
        dlat_g2.ravel(), dlon_g2.ravel(),
        (dlat_g2 * dlon_g2).ravel(),
        (dlat_g2**2).ravel(), (dlon_g2**2).ravel(),
        (dlat_g2**2 * dlon_g2).ravel(),
        (dlat_g2 * dlon_g2**2).ravel(),
        (dlat_g2**3).ravel(), (dlon_g2**3).ravel()
    ])
    A_grid_2m_norm = A_grid_2m.copy()
    A_grid_2m_norm[:, 1:] = (A_grid_2m[:, 1:] - col_means) / col_stds
    trend_2m = (A_grid_2m_norm @ coef_norm).reshape(ny_kr, nx_kr)

    # 2' → 1' (双线性) → 15" (双线性)
    kr_to_lr = (ny_lr / ny_kr, nx_lr / nx_kr)
    trend_1m = sp_zoom(trend_2m, kr_to_lr, order=1)
    z_krig_1m = sp_zoom(z_krig_2m, kr_to_lr, order=1)
    gravity_krig = sp_zoom(trend_1m + z_krig_1m, (upscale, upscale), order=1)
    return gravity_krig


def main(lon_range=(132, 152), lat_range=(26, 42), upscale=4):
    print("=" * 70)
    print("  M2 vs 传统方法 — 大区域精度对比 (20°×16°)")
    print("=" * 70)
    t_start = time.time()

    # === 数据加载 ===
    print(f"\n[1/4] 加载数据 (区域 lon={lon_range}, lat={lat_range})...")
    gravity_lr, _, lon_lr, lat_lr = load_ssv_region(lon_range, lat_range, load_vgg=False)
    ny_lr, nx_lr = gravity_lr.shape
    ny_hr, nx_hr = ny_lr * upscale, nx_lr * upscale
    lon_hr = np.linspace(lon_range[0], lon_range[1], nx_hr)
    lat_hr = np.linspace(lat_range[0], lat_range[1], ny_hr)
    print(f"  SSV 1': {gravity_lr.shape}  目标 15'': ({ny_hr}, {nx_hr}) "
          f"({ny_hr*nx_hr/1e6:.1f}M 像素)")

    train_points, val_points = load_all_cruises_split(lon_range, lat_range, VAL_CRUISE)
    print(f"  训练: {len(train_points):,} 点  验证 MR22-04: {len(val_points):,} 点")

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

    print("  加载 GEBCO 15'' 水深...")
    bathy_15s = load_bathy_15s(lon_range, lat_range, lon_hr, lat_hr)
    print(f"  GEBCO 水深: range=[{bathy_15s.min():.0f}, {bathy_15s.max():.0f}] m")

    region_size = lat_range[1] - lat_range[0]
    val_truth = val_points[:, 2]

    # === 构建 5 个模型 ===
    print("\n[2/4] 构建 5 个模型...")
    models = {}
    times = {}

    # 1. 双线性
    t0 = time.time()
    models['双线性插值'] = ssv_hr.copy()
    times['双线性插值'] = time.time() - t0
    print(f"  双线性: {times['双线性插值']:.1f}s")

    # 2. 频域融合
    t0 = time.time()
    print("\n  频域融合...")
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
    models['频域融合'] = ssv_lp + ship_hp
    times['频域融合'] = time.time() - t0
    print(f"  频域融合: {times['频域融合']:.1f}s")

    # 3. 克里金 + 趋势面
    t0 = time.time()
    print("\n  克里金 + 趋势面...")
    try:
        models['克里金+趋势面'] = kriging_with_external_train(
            gravity_lr=gravity_lr, ship_grav=ship_grav, train_mask=train_mask,
            lon_hr=lon_hr, lat_hr=lat_hr, lon_range=lon_range, lat_range=lat_range,
            upscale=upscale)
        times['克里金+趋势面'] = time.time() - t0
        print(f"  克里金: {times['克里金+趋势面']:.1f}s")
    except Exception as e:
        print(f"  克里金失败: {e}")
        models['克里金+趋势面'] = None
        times['克里金+趋势面'] = time.time() - t0

    # 4. 原始 KF (双线性先验) — 仅作参考, 与双线性有继承关系
    t0 = time.time()
    print("\n  原始 KF (SSV 双线性先验, 仅参考)...")
    r_orig = KalmanGravityReconstructor(
        grid_shape=(ny_hr, nx_hr),
        region_size_deg=region_size,
        lr_grid_shape=(ny_lr, nx_lr))
    r_orig.initialize(
        gravity_lr=gravity_lr, gravity_ship=ship_grav,
        ship_train_mask=train_mask,
        process_noise_std=2.0, sat_noise_std=3.0, ship_noise_std=1.0)
    models['原始KF(双线性先验)'] = r_orig.run(n_iterations=6, smooth_sigma=15.0)
    times['原始KF(双线性先验)'] = time.time() - t0
    print(f"  原始 KF: {times['原始KF(双线性先验)']:.1f}s")

    # 5. M2 地形先验 KF
    t0 = time.time()
    print("\n  M2 地形先验 KF (α=0.5)...")
    m2 = BathyPriorKF(
        hr_grid_shape=(ny_hr, nx_hr), lr_grid_shape=(ny_lr, nx_lr),
        region_size_deg=region_size, alpha=0.5)
    models['M2 地形先验KF'] = m2.build(
        gravity_lr=gravity_lr, ship_15s=ship_grav, train_mask=train_mask,
        lon_hr=lon_hr, lat_hr=lat_hr, lon_range=lon_range, lat_range=lat_range,
        upscale=upscale, bathy_15s=bathy_15s)
    times['M2 地形先验KF'] = time.time() - t0
    print(f"  M2: {times['M2 地形先验KF']:.1f}s")

    # 6. 纯 KF (零初值, 不用 SSV 双线性) — 与双线性最干净的对比
    t0 = time.time()
    print("\n  纯 KF (零初值, 不继承双线性)...")
    pkf = PureKF(
        hr_grid_shape=(ny_hr, nx_hr), lr_grid_shape=(ny_lr, nx_lr),
        region_size_deg=region_size)
    models['纯KF(零初值)'] = pkf.build(
        gravity_lr=gravity_lr, ship_15s=ship_grav, train_mask=train_mask,
        lon_hr=lon_hr, lat_hr=lat_hr, lon_range=lon_range, lat_range=lat_range,
        upscale=upscale, n_iter=8)
    times['纯KF(零初值)'] = time.time() - t0
    print(f"  纯 KF: {times['纯KF(零初值)']:.1f}s")

    # === 评估 ===
    print(f"\n[3/4] 独立验证 (MR22-04, n={len(val_points):,})...")
    method_names = [n for n in models.keys() if models[n] is not None]
    predictions = [get_predictions_at_val(models[n], val_points, lon_hr, lat_hr)
                   for n in method_names]

    print("\n  " + "=" * 92)
    print(f"  {'Method':<22} {'RMSE':<10} {'MAE':<10} {'Max|e|':<10} "
          f"{'STD':<10} {'Corr':<10} {'Bias':<10} {'Time(s)':<8}")
    print("  " + "-" * 92)
    metrics_rows = []
    bi_rmse = None
    for name, pred in zip(method_names, predictions):
        m = compute_metrics(pred, val_truth)
        m['time'] = times[name]
        metrics_rows.append((name, m))
        if name == '双线性插值':
            bi_rmse = m['RMSE']
        print(f"  {name:<22} {m['RMSE']:<10.2f} {m['MAE']:<10.2f} {m['Max|e|']:<10.2f} "
              f"{m['STD']:<10.2f} {m['Corr']:<10.4f} {m['Bias']:<10.2f} {times[name]:<8.1f}")

    # === 可视化 ===
    print("\n[4/4] 生成对比图...")
    n_methods = len(method_names)
    # 3 行 × 3 列: 6 个方法场图 + 1 个柱状图 + 2 个空白
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), sharex=True, sharey=True)
    axes = axes.ravel()

    all_vals = np.concatenate([models[n].ravel() for n in method_names])
    all_vals = all_vals[~np.isnan(all_vals)]
    vmax = float(np.percentile(np.abs(all_vals), 99.5))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    extent = [lon_hr[0], lon_hr[-1], lat_hr[0], lat_hr[-1]]

    # 找最优 (排除克里金这种失败的, 排除原始 KF 因其继承双线性)
    candidates = [(n, m) for n, m in metrics_rows
                    if '克里金' not in n and '原始KF' not in n and n != '双线性插值']
    best_name, best_m = min(candidates, key=lambda x: x[1]['RMSE'])

    for ax, (name, m) in zip(axes, metrics_rows):
        is_best = (name == best_name)
        is_bi = (name == '双线性插值')
        im = ax.imshow(models[name], cmap='RdBu_r', norm=norm, aspect='equal',
                       origin='lower', extent=extent, interpolation='nearest')
        ax.plot(val_points[:, 0], val_points[:, 1], 'k-', lw=0.3, alpha=0.5)
        ax.scatter(train_points[:, 0], train_points[:, 1],
                   s=0.15, c='yellow', alpha=0.2, edgecolors='none')
        ax.set_xlim(lon_range)
        ax.set_ylim(lat_range)
        ax.set_xlabel('经度 (°E)', fontsize=9)
        ax.set_ylabel('纬度 (°N)', fontsize=9)
        if bi_rmse is not None:
            impr = (1 - m['RMSE'] / bi_rmse) * 100
        else:
            impr = 0
        title = f'{name}\nRMSE={m["RMSE"]:.2f}'
        if is_best:
            title += f' ★最优 ({impr:+.1f}%)'
        elif is_bi:
            title += ' (基线)'
        else:
            title += f' ({impr:+.1f}%)'
        ax.set_title(title, fontsize=10,
                     fontweight='bold' if is_best else 'normal')
        ax.tick_params(labelsize=8)

    # 第 7 个面板: RMSE 柱状图
    ax_bar = axes[6]
    kf_names = [n for n, _ in metrics_rows if n != '双线性插值']
    rmses = [m['RMSE'] for n, m in metrics_rows if n != '双线性插值']
    if bi_rmse is None:
        bi_rmse = rmses[0]
    y_pos = np.arange(len(kf_names))
    colors = ['#A6BDDB'] * len(kf_names)
    if best_name in kf_names:
        best_idx = kf_names.index(best_name)
        colors[best_idx] = '#3690C0'
    bars = ax_bar.barh(y_pos, rmses, color=colors, edgecolor='none', height=0.55)
    ax_bar.axvline(bi_rmse, color='red', linestyle='--', lw=1.2,
                    label=f'双线性 RMSE={bi_rmse:.2f}')
    ax_bar.invert_yaxis()
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(kf_names, fontsize=8.5)
    ax_bar.set_xlabel('RMSE (mGal)', fontsize=9)
    ax_bar.set_title('各方法 RMSE 对比', fontsize=10, fontweight='bold')
    for i, (v, bar) in enumerate(zip(rmses, bars)):
        ax_bar.text(v + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{v:.2f}', va='center', fontsize=8.5,
                    fontweight='bold' if i == best_idx else 'normal')
    ax_bar.legend(fontsize=8, loc='lower right')
    for spine in ['top', 'right', 'left']:
        ax_bar.spines[spine].set_visible(False)
    ax_bar.grid(axis='x', alpha=0.3)

    # 第 8 个面板: 关键说明文字
    ax_txt = axes[7]
    ax_txt.axis('off')
    info_text = (
        "对比说明\n"
        "────────────────\n"
        "• 双线性: SSV 1' 双线性上采样\n"
        "  (无船载, 基线)\n\n"
        "• 频域融合: SSV 低频 + 船载高频\n"
        "  (Butterworth 滤波)\n\n"
        "• 克里金: 趋势面 + 残差克里金\n"
        "  (大区域失败)\n\n"
        "• 原始 KF: SSV 双线性先验 + KF\n"
        "  (与双线性有继承, 仅参考)\n\n"
        "• M2 地形先验: SSV + 地形 + KF\n"
        "  (含双线性成分)\n\n"
        "• 纯 KF: 零初值 + 卫星块约束 + 船载\n"
        "  (★ 与双线性最干净的对比)"
    )
    ax_txt.text(0.05, 0.95, info_text, transform=ax_txt.transAxes,
                fontsize=9, va='top', ha='left', family='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F0F0',
                          edgecolor='#CCCCCC', alpha=0.95))

    cb = fig.colorbar(im, ax=axes[:6].tolist(), orientation='vertical',
                      pad=0.015, shrink=0.85, aspect=35, extend='both')
    cb.set_label('FAA (mGal)', fontsize=10)

    fig.suptitle('KF 方法 vs 传统方法 — 大区域 (20°×16°) 重力场对比\n'
                 '黑线=MR22-04 验证航迹, 黄点=训练点  |  '
                 '纯 KF (零初值) 是与双线性最干净的对比',
                 fontsize=12, y=1.00)
    out_png = OUTPUT_DIR / "m2_vs_traditional.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  对比图: {out_png}")

    # 数值表
    out_txt = OUTPUT_DIR / "m2_vs_traditional_metrics.txt"
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 92 + "\n")
        f.write("  M2 vs 传统方法 — 大区域精度对比 (20°×16°)\n")
        f.write(f"  区域: lon={lon_range}, lat={lat_range}\n")
        f.write(f"  验证: MR22-04 (n={len(val_points):,}, 完全独立未参与训练)\n")
        f.write(f"  训练点: {len(train_points):,} (15'' 网格 {train_mask.sum():,} 像素)\n")
        f.write("=" * 92 + "\n")
        f.write(f"  {'Method':<22} {'RMSE':<10} {'MAE':<10} {'Max|e|':<10} "
                f"{'STD':<10} {'Corr':<10} {'Bias':<10} {'Impr%':<8} {'Time(s)':<8}\n")
        f.write("-" * 92 + "\n")
        for name, m in metrics_rows:
            impr = (1 - m['RMSE'] / bi_rmse) * 100 if bi_rmse else 0
            f.write(f"  {name:<22} {m['RMSE']:<10.2f} {m['MAE']:<10.2f} "
                    f"{m['Max|e|']:<10.2f} {m['STD']:<10.2f} {m['Corr']:<10.4f} "
                    f"{m['Bias']:<10.2f} {impr:<+8.1f} {m['time']:<8.1f}\n")
        f.write("=" * 92 + "\n")
        f.write(f"  最优: {best_name}  RMSE={best_m['RMSE']:.2f}  "
                f"改善={(1-best_m['RMSE']/bi_rmse)*100:.1f}%\n")
    print(f"  指标表: {out_txt}")

    print(f"\n  总用时: {time.time()-t_start:.0f}s")
    return metrics_rows


if __name__ == "__main__":
    main()
