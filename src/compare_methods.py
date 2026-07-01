"""三种方法重建 15" 重力场对比: KF vs 普通克里金+趋势 vs 频域融合。

在相同 10°×10° 区域、相同 80/20 船载划分、相同 15" 网格上对比 RMSE。

输出:
  - output_real/compare_gravity_15s.png  (4 子图: SSV / Bilinear / KF / Kriging / Fused)
  - output_real/compare_methods_table.txt (RMSE 表)
  - output_real/compare_methods_summary.txt
"""
import sys, time, json
from pathlib import Path
import numpy as np
from scipy.ndimage import zoom as sp_zoom, gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from run_pipeline import load_ssv_region, load_shipborne_gravity
from utils import split_shipborne_mask, compute_metrics

OUTPUT_DIR = Path(__file__).parent.parent / "output_real"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _save_gravity_png(fields, titles, save_path, vmin=None, vmax=None):
    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, data, title in zip(axes, fields, titles):
        d = np.asarray(data, dtype=float)
        im = ax.imshow(d, cmap="RdYlBu_r", aspect="auto", origin="lower",
                       vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("X (15'' pixel)")
        ax.set_ylabel("Y (15'' pixel)")
        plt.colorbar(im, ax=ax, label="mGal", shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def compare_methods(lon_range=(140, 150), lat_range=(30, 40), upscale=4):
    print("=" * 70)
    print("  Three-way comparison: KF vs Kriging vs Freq Fusion (15'' grid)")
    print("=" * 70)
    print(f"  Region: lon=[{lon_range[0]}, {lon_range[1]}], lat=[{lat_range[0]}, {lat_range[1]}]")
    print(f"  Grid: 1' → 15'' (upscale={upscale})")

    # === Step 1: Load data once ===
    print("\n[1/4] Loading SSV + shipborne (shared by all methods)...")
    gravity_lr, _, lon_lr, lat_lr = load_ssv_region(lon_range, lat_range, load_vgg=False)
    ny_lr, nx_lr = gravity_lr.shape
    ny_hr, nx_hr = ny_lr * upscale, nx_lr * upscale
    lon_hr = np.linspace(lon_range[0], lon_range[1], nx_hr)
    lat_hr = np.linspace(lat_range[0], lat_range[1], ny_hr)
    print(f"  SSV shape: {gravity_lr.shape},  Target 15'' grid: ({ny_hr}, {nx_hr})")

    ship_grav, _ = load_shipborne_gravity(lon_hr, lat_hr)

    # Residual → FAA using EGM2008 trend (same as run_pipeline)
    from kriging_baseline import _residual_to_faa
    ship_grav = _residual_to_faa(ship_grav, lon_range, lat_range, lon_lr, lat_lr, gravity_lr, upscale)
    n_ship = np.sum(~np.isnan(ship_grav))
    print(f"  Shipborne points: {n_ship:,}")

    ship_train_mask, ship_val_mask = split_shipborne_mask(ship_grav, validation_fraction=0.2)
    print(f"  Split: train={ship_train_mask.sum():,}, val={ship_val_mask.sum():,}")

    gravity_bilinear = sp_zoom(gravity_lr, (upscale, upscale), order=1)

    results = {}

    # === Step 2: Method A — KF (run_pipeline) ===
    print("\n[2/4] Method A: Kalman Filter (本项目方法)...")
    t0 = time.time()
    from kalman_gravity import KalmanGravityReconstructor
    reconstructor = KalmanGravityReconstructor(
        grid_shape=(ny_hr, nx_hr), region_size_deg=lat_range[1] - lat_range[0],
        lr_grid_shape=(ny_lr, nx_lr))
    reconstructor.initialize(
        gravity_lr=gravity_lr, gravity_ship=ship_grav,
        ship_train_mask=ship_train_mask,
        process_noise_std=2.0, sat_noise_std=3.0, ship_noise_std=1.0)
    gravity_kf = reconstructor.run(n_iterations=6, smooth_sigma=15.0)
    m_kf = compute_metrics(gravity_kf, ship_grav, mask=ship_val_mask)
    m_kf_bi = compute_metrics(gravity_bilinear, ship_grav, mask=ship_val_mask)
    print(f"  KF done in {time.time()-t0:.0f}s | RMSE={m_kf['RMSE']:.2f} mGal")
    results['KF'] = {'gravity': gravity_kf, 'metrics': m_kf}

    # === Step 3: Method B — Kriging + Trend (on 1' grid, upsampled to 15") ===
    # Per ggad471: kriging is performed on the altimetry model grid (1').
    # 15" output is obtained by bilinear upsampling — this is consistent with
    # ggad471 which itself only produces 1' nominal grids.
    print("\n[3/4] Method B: Ordinary Kriging + Trend surface (1' grid → 15'' bilinear)...")
    t0 = time.time()
    from kriging_baseline import _subsample_train_points
    try:
        from pykrige.ok import OrdinaryKriging

        lon0 = lon_range[0]
        lat0 = lat_range[0]
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
        rng_a = 0.25  # ~27 km
        OK = OrdinaryKriging(
            lon_pts, lat_pts, resid_train,
            variogram_model='spherical',
            variogram_parameters={'nugget': nugget, 'range': rng_a, 'sill': sill},
            verbose=False, enable_plotting=False,
            exact_values=False, pseudo_inv=True)
        print(f"  Kriging matrix built ({len(lon_pts)} pts) in {time.time()-t0:.1f}s")

        # Execute on 1' grid (600×600 = 360k targets, manageable in one batch)
        z_krig_1m, _ = OK.execute('grid', lon_lr, lat_lr, backend='vectorized')
        z_krig_1m = np.asarray(z_krig_1m)
        if z_krig_1m.shape != (ny_lr, nx_lr):
            z_krig_1m = z_krig_1m.T
        print(f"  Kriging 1' grid done in {time.time()-t0:.1f}s, upsampling to 15''...")

        # Trend on 1' grid
        ii_1m, jj_1m = np.meshgrid(np.arange(ny_lr), np.arange(nx_lr), indexing='ij')
        dlat_g1 = lat_lr[ii_1m] - lat0
        dlon_g1 = lon_lr[jj_1m] - lon0
        A_grid_1m = np.column_stack([np.ones(ny_lr * nx_lr), dlat_g1.ravel(), dlon_g1.ravel(),
                                      (dlat_g1 * dlon_g1).ravel(),
                                      (dlat_g1**2).ravel(), (dlon_g1**2).ravel()])
        trend_1m = (A_grid_1m @ coef).reshape(ny_lr, nx_lr)
        gravity_krig_1m = trend_1m + z_krig_1m

        # Bilinear upsample to 15"
        gravity_krig = sp_zoom(gravity_krig_1m, (upscale, upscale), order=1)
        m_krig = compute_metrics(gravity_krig, ship_grav, mask=ship_val_mask)
        print(f"  Kriging done in {time.time()-t0:.0f}s | RMSE={m_krig['RMSE']:.2f} mGal")
        results['Kriging+Trend'] = {'gravity': gravity_krig, 'metrics': m_krig}
    except Exception as e:
        import traceback
        print(f"  Kriging FAILED: {e}")
        traceback.print_exc()
        results['Kriging+Trend'] = None

    # === Step 4: Method C — Frequency-domain Fusion ===
    print("\n[4/4] Method C: Frequency-domain Fusion (Butterworth)...")
    t0 = time.time()
    from freq_fusion_baseline import butterworth_2d_filter

    ssv_hr = gravity_bilinear.copy()
    ship_train_only = np.full_like(ship_grav, np.nan)
    ship_train_only[ship_train_mask] = ship_grav[ship_train_mask]
    nan_mask = np.isnan(ship_train_only)
    ship_filled = ship_train_only.copy()
    ship_filled[nan_mask] = ssv_hr[nan_mask]
    ship_filled = gaussian_filter(ship_filled, sigma=2.0)
    ssv_hr_smooth = gaussian_filter(ssv_hr, sigma=2.0)

    dx_deg = (lon_range[1] - lon_range[0]) / nx_hr
    dy_deg = (lat_range[1] - lat_range[0]) / ny_hr
    lat_center = (lat_range[0] + lat_range[1]) / 2.0
    # Use ggad471 recommended cutoff (27 km for Japan region)
    # Coherence estimation fails here because SSV-filling makes ship & SSV
    # artificially coherent across all bands.
    cutoff_km = 27.0

    ship_hp = butterworth_2d_filter(ship_filled, dx_deg, dy_deg, lat_center,
                                     cutoff_km, btype='high', order=4)
    ssv_lp = butterworth_2d_filter(ssv_hr_smooth, dx_deg, dy_deg, lat_center,
                                    cutoff_km, btype='low', order=4)
    gravity_fused = ssv_lp + ship_hp
    m_fused = compute_metrics(gravity_fused, ship_grav, mask=ship_val_mask)
    print(f"  Freq fusion done in {time.time()-t0:.0f}s | RMSE={m_fused['RMSE']:.2f} mGal | cutoff={cutoff_km:.1f}km")
    results['FreqFusion'] = {'gravity': gravity_fused, 'metrics': m_fused, 'cutoff_km': cutoff_km}

    # === Summary ===
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY  (validation on 20% held-out shipborne points)")
    print("=" * 70)
    print(f"  {'Method':<22} {'RMSE (mGal)':<14} {'MAE (mGal)':<12} {'Corr':<8} {'Bias (mGal)':<12} {'N':<10}")
    print("  " + "-" * 78)
    print(f"  {'Bilinear (SSV upsamp)':<22} {m_kf_bi['RMSE']:<14.2f} {m_kf_bi['MAE']:<12.2f} "
          f"{m_kf_bi['Correlation']:<8.4f} {m_kf_bi['Bias']:<12.2f} {m_kf_bi['N']:<10,}")
    for name, r in results.items():
        if r is None:
            print(f"  {name:<22} {'FAILED':<14}")
            continue
        m = r['metrics']
        print(f"  {name:<22} {m['RMSE']:<14.2f} {m['MAE']:<12.2f} "
              f"{m['Correlation']:<8.4f} {m['Bias']:<12.2f} {m['N']:<10,}")

    # Improvement vs bilinear
    print("\n  Improvement over bilinear baseline:")
    for name, r in results.items():
        if r is None:
            continue
        impr = (1 - r['metrics']['RMSE'] / m_kf_bi['RMSE']) * 100
        print(f"    {name:<22}: {impr:+.2f}%")

    # Save outputs
    print("\n  Saving outputs...")
    np.save(OUTPUT_DIR / "compare_gravity_kf.npy", results['KF']['gravity'])
    if results['Kriging+Trend']:
        np.save(OUTPUT_DIR / "compare_gravity_kriging.npy", results['Kriging+Trend']['gravity'])
    np.save(OUTPUT_DIR / "compare_gravity_fused.npy", results['FreqFusion']['gravity'])
    np.save(OUTPUT_DIR / "compare_gravity_bilinear.npy", gravity_bilinear)
    np.save(OUTPUT_DIR / "compare_ship_val_mask.npy", ship_val_mask)

    fields = [gravity_bilinear, results['KF']['gravity']]
    titles = ["Bilinear (SSV 1'→15'')", "KF (本项目)"]
    if results['Kriging+Trend']:
        fields.append(results['Kriging+Trend']['gravity'])
        titles.append("Kriging+Trend")
    fields.append(results['FreqFusion']['gravity'])
    titles.append("Freq Fusion (Butterworth)")
    _save_gravity_png(fields, titles, OUTPUT_DIR / "compare_gravity_15s.png")

    # Summary text
    summary_lines = [
        "=" * 70,
        "  15'' Gravity Field Reconstruction — Three-way Comparison",
        "=" * 70,
        f"  Region: lon=[{lon_range[0]}, {lon_range[1]}], lat=[{lat_range[0]}, {lat_range[1]}]",
        f"  Grid: 1' SSV → 15'' (upscale={upscale})",
        f"  Shipborne points: {n_ship:,}  (train={ship_train_mask.sum():,}, val={ship_val_mask.sum():,})",
        f"  Validation: 20% held-out shipborne points (seed=42)",
        "",
        f"  {'Method':<22} {'RMSE':<10} {'MAE':<10} {'Corr':<8} {'Bias':<10} {'N':<10}",
        "  " + "-" * 70,
        f"  {'Bilinear':<22} {m_kf_bi['RMSE']:<10.2f} {m_kf_bi['MAE']:<10.2f} "
        f"{m_kf_bi['Correlation']:<8.4f} {m_kf_bi['Bias']:<10.2f} {m_kf_bi['N']:<10,}",
    ]
    for name, r in results.items():
        if r is None:
            summary_lines.append(f"  {name:<22} FAILED")
            continue
        m = r['metrics']
        summary_lines.append(
            f"  {name:<22} {m['RMSE']:<10.2f} {m['MAE']:<10.2f} "
            f"{m['Correlation']:<8.4f} {m['Bias']:<10.2f} {m['N']:<10,}")
    summary_lines.append("")
    summary_lines.append("  Improvement over bilinear baseline:")
    for name, r in results.items():
        if r is None:
            continue
        impr = (1 - r['metrics']['RMSE'] / m_kf_bi['RMSE']) * 100
        summary_lines.append(f"    {name:<22}: {impr:+.2f}%")

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    with open(OUTPUT_DIR / "compare_methods_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"\n  Summary saved: {OUTPUT_DIR / 'compare_methods_summary.txt'}")

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
    compare_methods(
        lon_range=(args.lon_min, args.lon_max),
        lat_range=(args.lat_min, args.lat_max),
        upscale=args.upscale)
