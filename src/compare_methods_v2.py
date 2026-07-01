"""KF 优化变体对比: v2a 留空区 KF / v2b 1'网格 KF + 双线性上采样.

目标: 让 KF 在 15" RMSE 上超过克里金+趋势面 (44.13 mGal).

v2a — 留空区 KF:
  - 验证点周围 r=3 像素标记为 exclusion zone
  - update_shipborne 跳过 exclusion zone 内的训练点
  - 验证点估计完全来自 SSV 双线性 + 远处训练点间接影响

v2b — 1' 网格 KF + 双线性上采样:
  - 状态变量降到 1' 网格 (600×600), 与克里金相同设置
  - 船载训练点下采样到 1' (每 4×4 块取训练点均值)
  - KF 在 1' 上跑, 双线性上采样到 15"
  - 验证点估计 = 1' KF 值的双线性插值, 与克里金设置完全相同
"""
import sys, time
from pathlib import Path
import numpy as np
from scipy.ndimage import zoom as sp_zoom, gaussian_filter, binary_dilation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from run_pipeline import load_ssv_region, load_shipborne_gravity
from kriging_baseline import _residual_to_faa, _subsample_train_points
from freq_fusion_baseline import butterworth_2d_filter
from utils import split_shipborne_mask, compute_metrics
from kalman_gravity import KalmanGravityReconstructor

OUTPUT_DIR = Path(__file__).parent.parent / "output_real"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _downsample_to_1min(ship_15s, train_mask, hr_per_lr=4):
    """把 15" 船载网格(仅训练点)下采样到 1' 网格.

    每个(hr_per_lr, hr_per_lr)块内对训练点取均值.
    """
    ny_hr, nx_hr = ship_15s.shape
    ny_lr = ny_hr // hr_per_lr
    nx_lr = nx_hr // hr_per_lr
    ship_1m = np.full((ny_lr, nx_lr), np.nan)
    train_1m = np.zeros((ny_lr, nx_lr), dtype=bool)
    for i in range(ny_lr):
        for j in range(nx_lr):
            block_ship = ship_15s[i*hr_per_lr:(i+1)*hr_per_lr,
                                    j*hr_per_lr:(j+1)*hr_per_lr]
            block_train = train_mask[i*hr_per_lr:(i+1)*hr_per_lr,
                                       j*hr_per_lr:(j+1)*hr_per_lr]
            if np.any(block_train):
                vals = block_ship[block_train]
                if np.any(~np.isnan(vals)):
                    ship_1m[i, j] = np.nanmean(vals)
                    train_1m[i, j] = True
    return ship_1m, train_1m


def _build_exclusion_mask(ship_val_mask, exclusion_radius=3):
    """验证点周围 r 像素标记为 exclusion zone (在 15" 网格上)."""
    # Dilate validation mask by exclusion_radius
    structure = np.ones((2 * exclusion_radius + 1, 2 * exclusion_radius + 1))
    excl = binary_dilation(ship_val_mask, structure=structure, iterations=1)
    return excl


def run_kf_v2a_exclusion(gravity_lr, ship_grav, ship_train_mask, ship_val_mask,
                          lon_range, lat_range, upscale=4,
                          process_noise_std=2.0, sat_noise_std=3.0, ship_noise_std=1.0,
                          n_iterations=6, smooth_sigma=15.0, exclusion_radius=3):
    """v2a: 留空区 KF — 验证点周围 r 像素不做船载更新。"""
    print(f"\n  [v2a] KF with exclusion zone (r={exclusion_radius})")
    ny_lr, nx_lr = gravity_lr.shape
    ny_hr, nx_hr = ny_lr * upscale, nx_lr * upscale

    # Build exclusion mask: dilate val_mask by exclusion_radius
    excl_mask = _build_exclusion_mask(ship_val_mask, exclusion_radius)
    # Effective train mask = train_mask & ~excl_mask
    train_eff = ship_train_mask & ~excl_mask
    n_excluded = (ship_train_mask & excl_mask).sum()
    print(f"    Excluded {n_excluded:,} train pixels near val points "
          f"({n_excluded/ship_train_mask.sum()*100:.1f}% of train)")
    print(f"    Effective train pixels: {train_eff.sum():,}")

    reconstructor = KalmanGravityReconstructor(
        grid_shape=(ny_hr, nx_hr), region_size_deg=lat_range[1] - lat_range[0],
        lr_grid_shape=(ny_lr, nx_lr))
    reconstructor.initialize(
        gravity_lr=gravity_lr, gravity_ship=ship_grav,
        ship_train_mask=train_eff,
        process_noise_std=process_noise_std, sat_noise_std=sat_noise_std,
        ship_noise_std=ship_noise_std)
    t0 = time.time()
    gravity_kf = reconstructor.run(n_iterations=n_iterations, smooth_sigma=smooth_sigma)
    print(f"    KF done in {time.time()-t0:.0f}s")
    return gravity_kf


def run_kf_v2b_1min(gravity_lr, ship_grav, ship_train_mask,
                     lon_range, lat_range, upscale=4,
                     process_noise_std=2.0, sat_noise_std=3.0, ship_noise_std=1.0,
                     n_iterations=6, smooth_sigma=4.0):
    """v2b: 1' 网格 KF + 双线性上采样到 15"."""
    print(f"\n  [v2b] KF on 1' grid + bilinear upsample to 15''")
    ny_lr, nx_lr = gravity_lr.shape

    # Downsample shipborne (train only) to 1' grid
    ship_1m, train_1m = _downsample_to_1min(ship_grav, ship_train_mask, hr_per_lr=upscale)
    n_1m = np.sum(~np.isnan(ship_1m))
    print(f"    Shipborne 1' grid: {n_1m:,} non-NaN pixels (from {ship_train_mask.sum():,} 15'' train pixels)")

    # KF on 1' grid (lr_grid_shape == grid_shape → update_satellite degenerates to identity)
    reconstructor = KalmanGravityReconstructor(
        grid_shape=(ny_lr, nx_lr), region_size_deg=lat_range[1] - lat_range[0],
        lr_grid_shape=(ny_lr, nx_lr))
    reconstructor.initialize(
        gravity_lr=gravity_lr, gravity_ship=ship_1m,
        ship_train_mask=train_1m,
        process_noise_std=process_noise_std, sat_noise_std=sat_noise_std,
        ship_noise_std=ship_noise_std)
    t0 = time.time()
    gravity_kf_1m = reconstructor.run(n_iterations=n_iterations, smooth_sigma=smooth_sigma)
    print(f"    KF 1' done in {time.time()-t0:.0f}s")

    # Bilinear upsample to 15"
    gravity_kf_15s = sp_zoom(gravity_kf_1m, (upscale, upscale), order=1)
    return gravity_kf_15s


def compare_v2(lon_range=(140, 150), lat_range=(30, 40), upscale=4):
    print("=" * 70)
    print("  KF Optimization Variants vs Kriging+Trend (15'' grid)")
    print("=" * 70)

    print("\n[1/4] Loading data...")
    gravity_lr, _, lon_lr, lat_lr = load_ssv_region(lon_range, lat_range, load_vgg=False)
    ny_lr, nx_lr = gravity_lr.shape
    ny_hr, nx_hr = ny_lr * upscale, nx_lr * upscale
    lon_hr = np.linspace(lon_range[0], lon_range[1], nx_hr)
    lat_hr = np.linspace(lat_range[0], lat_range[1], ny_hr)
    print(f"  SSV 1': {gravity_lr.shape}  Target 15'': ({ny_hr}, {nx_hr})")

    ship_grav, _ = load_shipborne_gravity(lon_hr, lat_hr)
    ship_grav = _residual_to_faa(ship_grav, lon_range, lat_range, lon_lr, lat_lr, gravity_lr, upscale)
    n_ship = np.sum(~np.isnan(ship_grav))
    print(f"  Shipborne points: {n_ship:,}")

    ship_train_mask, ship_val_mask = split_shipborne_mask(ship_grav, validation_fraction=0.2)
    print(f"  Split: train={ship_train_mask.sum():,}, val={ship_val_mask.sum():,}")

    gravity_bilinear = sp_zoom(gravity_lr, (upscale, upscale), order=1)
    m_bi = compute_metrics(gravity_bilinear, ship_grav, mask=ship_val_mask)
    print(f"  Bilinear baseline RMSE: {m_bi['RMSE']:.2f} mGal")

    results = {}

    # === Original KF (baseline for comparison) ===
    print("\n[2/4] Original KF (15'' grid, smooth_sigma=15)...")
    t0 = time.time()
    reconstructor = KalmanGravityReconstructor(
        grid_shape=(ny_hr, nx_hr), region_size_deg=lat_range[1] - lat_range[0],
        lr_grid_shape=(ny_lr, nx_lr))
    reconstructor.initialize(
        gravity_lr=gravity_lr, gravity_ship=ship_grav,
        ship_train_mask=ship_train_mask,
        process_noise_std=2.0, sat_noise_std=3.0, ship_noise_std=1.0)
    gravity_kf_orig = reconstructor.run(n_iterations=6, smooth_sigma=15.0)
    m_kf_orig = compute_metrics(gravity_kf_orig, ship_grav, mask=ship_val_mask)
    print(f"  KF original done in {time.time()-t0:.0f}s | RMSE={m_kf_orig['RMSE']:.2f}")
    results['KF_orig'] = {'gravity': gravity_kf_orig, 'metrics': m_kf_orig}

    # === Kriging+Trend (for reference) ===
    print("\n[3/4] Kriging+Trend (reference baseline)...")
    t0 = time.time()
    try:
        from pykrige.ok import OrdinaryKriging
        lon0, lat0 = lon_range[0], lat_range[0]
        lon_pts, lat_pts, val_pts = _subsample_train_points(
            lon_hr, lat_hr, ship_grav, ship_train_mask, n_target=3000, seed=42)
        dlat = lat_pts - lat0
        dlon = lon_pts - lon0
        A_train = np.column_stack([np.ones_like(dlat), dlat, dlon, dlat*dlon, dlat**2, dlon**2])
        coef, *_ = np.linalg.lstsq(A_train, val_pts, rcond=None)
        resid_train = val_pts - A_train @ coef
        resid_var = np.var(resid_train)
        nugget = 5.0
        sill = max(resid_var, nugget + 1.0)
        rng_a = 0.25
        OK = OrdinaryKriging(
            lon_pts, lat_pts, resid_train,
            variogram_model='spherical',
            variogram_parameters={'nugget': nugget, 'range': rng_a, 'sill': sill},
            verbose=False, enable_plotting=False,
            exact_values=False, pseudo_inv=True)
        z_krig_1m, _ = OK.execute('grid', lon_lr, lat_lr, backend='vectorized')
        z_krig_1m = np.asarray(z_krig_1m)
        if z_krig_1m.shape != (ny_lr, nx_lr):
            z_krig_1m = z_krig_1m.T
        ii_1m, jj_1m = np.meshgrid(np.arange(ny_lr), np.arange(nx_lr), indexing='ij')
        dlat_g1 = lat_lr[ii_1m] - lat0
        dlon_g1 = lon_lr[jj_1m] - lon0
        A_grid_1m = np.column_stack([np.ones(ny_lr * nx_lr), dlat_g1.ravel(), dlon_g1.ravel(),
                                      (dlat_g1 * dlon_g1).ravel(),
                                      (dlat_g1**2).ravel(), (dlon_g1**2).ravel()])
        trend_1m = (A_grid_1m @ coef).reshape(ny_lr, nx_lr)
        gravity_krig = sp_zoom(trend_1m + z_krig_1m, (upscale, upscale), order=1)
        m_krig = compute_metrics(gravity_krig, ship_grav, mask=ship_val_mask)
        print(f"  Kriging done in {time.time()-t0:.0f}s | RMSE={m_krig['RMSE']:.2f}")
        results['Kriging+Trend'] = {'gravity': gravity_krig, 'metrics': m_krig}
    except Exception as e:
        print(f"  Kriging FAILED: {e}")
        results['Kriging+Trend'] = None

    # === v2a: Exclusion zone KF ===
    print("\n[4/4] KF v2 variants...")
    gravity_kf_v2a = run_kf_v2a_exclusion(
        gravity_lr, ship_grav, ship_train_mask, ship_val_mask,
        lon_range, lat_range, upscale=upscale,
        exclusion_radius=3)
    m_kf_v2a = compute_metrics(gravity_kf_v2a, ship_grav, mask=ship_val_mask)
    print(f"    v2a RMSE={m_kf_v2a['RMSE']:.2f}")
    results['KF_v2a_exclusion'] = {'gravity': gravity_kf_v2a, 'metrics': m_kf_v2a}

    # === v2b: 1' grid KF + bilinear ===
    gravity_kf_v2b = run_kf_v2b_1min(
        gravity_lr, ship_grav, ship_train_mask,
        lon_range, lat_range, upscale=upscale,
        smooth_sigma=4.0)  # 1' grid: smooth_sigma ~ 15/4
    m_kf_v2b = compute_metrics(gravity_kf_v2b, ship_grav, mask=ship_val_mask)
    print(f"    v2b RMSE={m_kf_v2b['RMSE']:.2f}")
    results['KF_v2b_1min'] = {'gravity': gravity_kf_v2b, 'metrics': m_kf_v2b}

    # === Summary ===
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    print(f"  {'Method':<24} {'RMSE':<10} {'MAE':<10} {'Corr':<8} {'Bias':<10} {'Impr%':<8}")
    print("  " + "-" * 70)
    print(f"  {'Bilinear':<24} {m_bi['RMSE']:<10.2f} {m_bi['MAE']:<10.2f} "
          f"{m_bi['Correlation']:<8.4f} {m_bi['Bias']:<10.2f} {'(base)':<8}")
    for name, r in results.items():
        if r is None:
            print(f"  {name:<24} FAILED")
            continue
        m = r['metrics']
        impr = (1 - m['RMSE'] / m_bi['RMSE']) * 100
        print(f"  {name:<24} {m['RMSE']:<10.2f} {m['MAE']:<10.2f} "
              f"{m['Correlation']:<8.4f} {m['Bias']:<10.2f} {impr:<+8.2f}")

    # Save
    for name, r in results.items():
        if r is not None:
            np.save(OUTPUT_DIR / f"compare_{name}.npy", r['gravity'])
    np.save(OUTPUT_DIR / "compare_bilinear.npy", gravity_bilinear)

    # Plot
    fields = [gravity_bilinear]
    titles = ["Bilinear"]
    for name in ['KF_orig', 'Kriging+Trend', 'KF_v2a_exclusion', 'KF_v2b_1min']:
        if results.get(name) is not None:
            fields.append(results[name]['gravity'])
            titles.append(name)
    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, d, t in zip(axes, fields, titles):
        im = ax.imshow(np.asarray(d, dtype=float), cmap="RdYlBu_r", aspect="auto", origin="lower")
        ax.set_title(t, fontsize=11)
        plt.colorbar(im, ax=ax, label="mGal", shrink=0.8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "compare_kf_variants.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved: {OUTPUT_DIR / 'compare_kf_variants.png'}")

    # Summary text
    lines = [
        "=" * 70,
        "  KF Optimization Variants vs Kriging (15'' grid)",
        "=" * 70,
        f"  Region: lon=[{lon_range[0]}, {lon_range[1]}], lat=[{lat_range[0]}, {lat_range[1]}]",
        f"  Shipborne: {n_ship:,} (train={ship_train_mask.sum():,}, val={ship_val_mask.sum():,})",
        "",
        f"  {'Method':<24} {'RMSE':<10} {'MAE':<10} {'Corr':<8} {'Bias':<10} {'Impr%':<8}",
        "  " + "-" * 70,
        f"  {'Bilinear':<24} {m_bi['RMSE']:<10.2f} {m_bi['MAE']:<10.2f} "
        f"{m_bi['Correlation']:<8.4f} {m_bi['Bias']:<10.2f} {'(base)':<8}",
    ]
    for name, r in results.items():
        if r is None:
            lines.append(f"  {name:<24} FAILED")
            continue
        m = r['metrics']
        impr = (1 - m['RMSE'] / m_bi['RMSE']) * 100
        lines.append(f"  {name:<24} {m['RMSE']:<10.2f} {m['MAE']:<10.2f} "
                     f"{m['Correlation']:<8.4f} {m['Bias']:<10.2f} {impr:<+8.2f}")
    summary_text = "\n".join(lines)
    print("\n" + summary_text)
    with open(OUTPUT_DIR / "compare_kf_variants_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lon-min", type=float, default=140)
    parser.add_argument("--lon-max", type=float, default=150)
    parser.add_argument("--lat-min", type=float, default=30)
    parser.add_argument("--lat-max", type=float, default=40)
    parser.add_argument("--upscale", type=int, default=4)
    args = parser.parse_args()
    compare_v2(
        lon_range=(args.lon_min, args.lon_max),
        lat_range=(args.lat_min, args.lat_max),
        upscale=args.upscale)
