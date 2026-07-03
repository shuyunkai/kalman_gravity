"""独立巡航验证 v2: 多尺度精度评估 + 与船载真值对比.

验证机制:
  - 训练: 30 个巡航原始点 → 精确像素赋值到 15" 网格
  - 验证: MR22-04 巡航 (完全独立, 未参与训练)

三个尺度对比模型 vs 船载:
  1. 15" 原始点验证: 模型 15" 值 vs 船载原始 10s 点
     (含代表性误差, 460m 像素 vs 50m 采样)
  2. 1' 网格验证: 模型下采样到 1' vs 船载 1' 块均值
     (论文标准, 与 ggad471/NSOAS24/SDUST 可直接对比)
  3. 分波段验证: 长波/中波/短波分别看 RMSE

船载重力本身精度 ~1-2 mGal (交叉点不符值), 视为真值.
模型与船载的 RMSE 即"模型相对真值的精度损失".
"""
import sys, time
from pathlib import Path
import numpy as np
from scipy.ndimage import zoom as sp_zoom, gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))

from run_pipeline import load_ssv_region, NC_ENGINE, DATA_DIR
from kalman_gravity import KalmanGravityReconstructor
from utils import compute_metrics

JP_DATA_DIR = Path(__file__).parent.parent / "jp_data"
OUTPUT_DIR = Path(__file__).parent.parent / "output_real"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_CRUISE = "MR22-04_gravity_dmo"


def load_cruise_points(cruise_dir, lon_range, lat_range):
    grv_files = list(cruise_dir.glob("*.grv"))
    if not grv_files:
        return None
    pts = []
    with open(grv_files[0], 'r', errors='ignore') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                lat = float(parts[2])
                lon = float(parts[3])
                faa = float(parts[5])
            except (ValueError, IndexError):
                continue
            if (lon_range[0] <= lon <= lon_range[1] and
                lat_range[0] <= lat <= lat_range[1] and
                abs(faa) < 5000):
                pts.append([lon, lat, faa])
    return np.array(pts, dtype=np.float64) if pts else None


def load_all_cruises_split(lon_range, lat_range, val_cruise=VAL_CRUISE):
    train_points, val_points = [], None
    n_train = 0
    for cruise_dir in sorted(JP_DATA_DIR.iterdir()):
        if not cruise_dir.is_dir():
            continue
        pts = load_cruise_points(cruise_dir, lon_range, lat_range)
        if pts is None or len(pts) == 0:
            continue
        if cruise_dir.name == val_cruise:
            val_points = pts
            print(f"    [VAL] {cruise_dir.name}: {len(pts):,} points")
        else:
            train_points.append(pts)
            n_train += 1
            print(f"    [train] {cruise_dir.name}: {len(pts):,} points")
    return np.vstack(train_points), val_points


def assign_points_to_grid(points, lon_grid, lat_grid):
    ny, nx = len(lat_grid), len(lon_grid)
    grid = np.full((ny, nx), np.nan)
    j_idx = np.round((points[:, 0] - lon_grid[0]) / (lon_grid[1] - lon_grid[0])).astype(int)
    i_idx = np.round((points[:, 1] - lat_grid[0]) / (lat_grid[1] - lat_grid[0])).astype(int)
    valid = (j_idx >= 0) & (j_idx < nx) & (i_idx >= 0) & (i_idx < ny)
    grid[i_idx[valid], j_idx[valid]] = points[valid, 2]
    return grid


def downsample_to_1min(ship_15s, hr_per_lr=4):
    ny_hr, nx_hr = ship_15s.shape
    ny_lr, nx_lr = ny_hr // hr_per_lr, nx_hr // hr_per_lr
    ship_1m = np.full((ny_lr, nx_lr), np.nan)
    train_1m = np.zeros((ny_lr, nx_lr), dtype=bool)
    for i in range(ny_lr):
        for j in range(nx_lr):
            block = ship_15s[i*hr_per_lr:(i+1)*hr_per_lr, j*hr_per_lr:(j+1)*hr_per_lr]
            valid = ~np.isnan(block)
            if np.any(valid):
                ship_1m[i, j] = np.nanmean(block)
                train_1m[i, j] = True
    return ship_1m, train_1m


def evaluate_15s_raw(gravity_15s, val_points, lon_hr, lat_hr):
    """15" 模型值 vs 船载原始 10s 点 (含代表性误差)."""
    interp = RegularGridInterpolator(
        (lat_hr, lon_hr), gravity_15s, bounds_error=False, fill_value=np.nan)
    pred = interp(np.column_stack([val_points[:, 1], val_points[:, 0]]))
    truth = val_points[:, 2]
    valid = ~(np.isnan(pred) | np.isnan(truth))
    m = compute_metrics(pred[valid], truth[valid])
    # Bias-corrected RMSE (扣除系统偏差, 反映真实精度)
    diff = pred[valid] - truth[valid]
    m['RMSE_bc'] = float(np.sqrt(np.mean((diff - diff.mean())**2)))
    m['STD'] = float(np.std(diff))
    return m


def evaluate_1min_grid(gravity_15s, val_points, lon_range, lat_range, upscale=4):
    """1' 网格验证: 模型下采样到 1' vs 船载 1' 块均值 (论文标准)."""
    ny_hr, nx_hr = gravity_15s.shape
    ny_lr, nx_lr = ny_hr // upscale, nx_hr // upscale
    # Model 15" → 1' by block mean
    model_1m = np.full((ny_lr, nx_lr), np.nan)
    for i in range(ny_lr):
        for j in range(nx_lr):
            block = gravity_15s[i*upscale:(i+1)*upscale, j*upscale:(j+1)*upscale]
            model_1m[i, j] = np.nanmean(block)
    # Ship points → 1' grid (block mean)
    lon_lr = np.linspace(lon_range[0], lon_range[1], nx_lr)
    lat_lr = np.linspace(lat_range[0], lat_range[1], ny_lr)
    ship_1m_sum = np.zeros((ny_lr, nx_lr))
    ship_1m_cnt = np.zeros((ny_lr, nx_lr), dtype=int)
    j_idx = np.round((val_points[:, 0] - lon_lr[0]) / (lon_lr[1] - lon_lr[0])).astype(int)
    i_idx = np.round((val_points[:, 1] - lat_lr[0]) / (lat_lr[1] - lat_lr[0])).astype(int)
    valid = (j_idx >= 0) & (j_idx < nx_lr) & (i_idx >= 0) & (i_idx < ny_lr)
    np.add.at(ship_1m_sum, (i_idx[valid], j_idx[valid]), val_points[valid, 2])
    np.add.at(ship_1m_cnt, (i_idx[valid], j_idx[valid]), 1)
    ship_1m_mean = np.where(ship_1m_cnt > 0, ship_1m_sum / np.maximum(ship_1m_cnt, 1), np.nan)

    # Compare where both are non-NaN
    mask = ~np.isnan(model_1m) & ~np.isnan(ship_1m_mean)
    m = compute_metrics(model_1m[mask], ship_1m_mean[mask])
    diff = model_1m[mask] - ship_1m_mean[mask]
    m['RMSE_bc'] = float(np.sqrt(np.mean((diff - diff.mean())**2)))
    m['STD'] = float(np.std(diff))
    return m, mask.sum()


def evaluate_bandpass(gravity_15s, val_points, lon_hr, lat_hr, lon_range, lat_range,
                      lam_min=None, lam_max=None):
    """分波段验证: 模型与船载点都做带通滤波后对比."""
    from utils import bandpass_filter
    dx_km = (lat_range[1] - lat_range[0]) * 111.32 / gravity_15s.shape[0]
    model_bp = bandpass_filter(gravity_15s, dx_km, lam_min=lam_min, lam_max=lam_max)
    interp = RegularGridInterpolator(
        (lat_hr, lon_hr), model_bp, bounds_error=False, fill_value=np.nan)
    pred = interp(np.column_stack([val_points[:, 1], val_points[:, 0]]))
    truth = val_points[:, 2]
    valid = ~(np.isnan(pred) | np.isnan(truth))
    # Subtract mean (bandpass should be zero-mean, but just in case)
    pred_v = pred[valid] - pred[valid].mean()
    truth_v = truth[valid] - truth[valid].mean()
    m = compute_metrics(pred_v, truth_v)
    m['RMSE_bc'] = m['RMSE']  # already bias-corrected
    return m


def apply_egm2008_trend_to_points(points, lon_range, lat_range):
    """对残余重力点应用 EGM2008 趋势恢复."""
    try:
        ds_egm = xr.open_dataset(
            DATA_DIR / "icgem" / "grid" / "EGM2008_gravity_anomaly.nc",
            engine=NC_ENGINE)
        vname = [v for v in ds_egm.variables if len(ds_egm[v].dims) >= 2][0]
        grav_egm = ds_egm[vname].sel(
            lon=slice(lon_range[0], lon_range[1]),
            lat=slice(lat_range[1], lat_range[0])).values
        ds_egm.close()
        egm_trend = gaussian_filter(grav_egm, sigma=10)
        xi_egm = np.linspace(lon_range[0], lon_range[1], egm_trend.shape[1])
        yi_egm = np.linspace(lat_range[1], lat_range[0], egm_trend.shape[0])
        interp = RegularGridInterpolator(
            (yi_egm, xi_egm), egm_trend, bounds_error=False, fill_value=np.nan)
        trend = interp(np.column_stack([points[:, 1], points[:, 0]]))
        points[:, 2] = points[:, 2] + trend
        return points
    except Exception as e:
        print(f"  EGM2008 trend failed: {e}")
        return points


def main(lon_range=(140, 150), lat_range=(30, 40), upscale=4):
    print("=" * 70)
    print(f"  Independent Cruise Validation v2 — Multi-scale vs Shipborne")
    print("=" * 70)
    print(f"  Region: lon=[{lon_range[0]}, {lon_range[1]}], lat=[{lat_range[0]}, {lat_range[1]}]")
    print(f"  Val cruise: {VAL_CRUISE} (fully independent, never used in training)")

    print("\n[1/5] Loading SSV33.1...")
    gravity_lr, _, lon_lr, lat_lr = load_ssv_region(lon_range, lat_range, load_vgg=False)
    ny_lr, nx_lr = gravity_lr.shape
    ny_hr, nx_hr = ny_lr * upscale, nx_lr * upscale
    lon_hr = np.linspace(lon_range[0], lon_range[1], nx_hr)
    lat_hr = np.linspace(lat_range[0], lat_range[1], ny_hr)
    print(f"  SSV 1': {gravity_lr.shape}  Target 15'': ({ny_hr}, {nx_hr})")

    print("\n[2/5] Loading cruises...")
    train_points, val_points = load_all_cruises_split(lon_range, lat_range, VAL_CRUISE)
    print(f"  Train: {len(train_points):,} raw points")
    print(f"  Val:   {len(val_points):,} raw points (MR22-04)")
    # Val points (MR22-04) are already FAA — no trend recovery needed
    # (verified separately: Bias=1.63, STD=3.86 mGal vs SSV, consistent with paper)

    print("\n[3/5] Assigning train points to 15'' grid...")
    ship_grav = assign_points_to_grid(train_points, lon_hr, lat_hr)
    n_ship = np.sum(~np.isnan(ship_grav))
    print(f"  Train 15'' pixels: {n_ship:,} ({n_ship/ship_grav.size*100:.2f}% coverage)")

    # Check datum: train points vs SSV at train points
    train_mask_raw = ~np.isnan(ship_grav)
    ssv_hr = sp_zoom(gravity_lr, (upscale, upscale), order=1)
    diff_train_ssv = ship_grav[train_mask_raw] - ssv_hr[train_mask_raw]
    print(f"  Train - SSV at train pixels: Bias={np.nanmean(diff_train_ssv):.2f} "
          f"STD={np.nanstd(diff_train_ssv):.2f} mGal")
    if abs(np.nanmean(diff_train_ssv)) < 5 and np.nanstd(diff_train_ssv) < 15:
        print("  → Train data already on FAA datum (same as SSV), no trend recovery needed")
        train_on_faa = ship_grav
    else:
        # Train data is residual → add SSV long-wave trend (NOT EGM2008) to match SSV datum
        print("  → Train data appears residual, adding SSV long-wave trend (sigma=10)")
        ssv_trend_lr = gaussian_filter(gravity_lr, sigma=10)
        ssv_trend_hr = sp_zoom(ssv_trend_lr, (upscale, upscale), order=1)
        train_on_faa = ship_grav.copy()
        train_on_faa[train_mask_raw] = ship_grav[train_mask_raw] + ssv_trend_hr[train_mask_raw]
        diff_after = train_on_faa[train_mask_raw] - ssv_hr[train_mask_raw]
        print(f"  After SSV trend recovery: Bias={np.nanmean(diff_after):.2f} "
              f"STD={np.nanstd(diff_after):.2f} mGal")

    ship_grav = train_on_faa
    train_mask = ~np.isnan(ship_grav)

    # Val points: MR22-04 is already FAA (verified: Bias=1.63, STD=3.86 vs SSV)
    # Do NOT apply any trend recovery
    val_mean = np.mean(val_points[:, 2])
    print(f"  Val points FAA mean: {val_mean:.2f} mGal (using as-is, no trend recovery)")

    print("\n[4/5] Reconstructing with 3 methods...")
    results = {}

    # Bilinear (传统基线)
    gravity_bilinear = sp_zoom(gravity_lr, (upscale, upscale), order=1)
    results['Bilinear'] = gravity_bilinear

    # KF original (本项目最优方法, 15" 网格)
    print("\n  [b] KF (15'' grid, 本项目方法)...")
    t0 = time.time()
    r = KalmanGravityReconstructor(
        grid_shape=(ny_hr, nx_hr),
        region_size_deg=lat_range[1] - lat_range[0],
        lr_grid_shape=(ny_lr, nx_lr))
    r.initialize(
        gravity_lr=gravity_lr, gravity_ship=ship_grav,
        ship_train_mask=train_mask,
        process_noise_std=2.0, sat_noise_std=3.0, ship_noise_std=1.0)
    results['KF'] = r.run(n_iterations=6, smooth_sigma=15.0)
    print(f"      done {time.time()-t0:.0f}s")

    # Freq Fusion (Butterworth, 传统方法对照)
    print("\n  [c] Frequency-domain Fusion (Butterworth, cutoff=27km)...")
    t0 = time.time()
    try:
        from freq_fusion_baseline import butterworth_2d_filter
        ssv_hr_smooth = gaussian_filter(results['Bilinear'], sigma=2.0)
        ship_filled = ship_grav.copy()
        nan_mask = np.isnan(ship_filled)
        ship_filled[nan_mask] = results['Bilinear'][nan_mask]
        ship_filled = gaussian_filter(ship_filled, sigma=2.0)
        dx_deg = (lon_range[1] - lon_range[0]) / nx_hr
        dy_deg = (lat_range[1] - lat_range[0]) / ny_hr
        lat_center = (lat_range[0] + lat_range[1]) / 2.0
        cutoff_km = 27.0
        ship_hp = butterworth_2d_filter(ship_filled, dx_deg, dy_deg, lat_center,
                                         cutoff_km, btype='high', order=4)
        ssv_lp = butterworth_2d_filter(ssv_hr_smooth, dx_deg, dy_deg, lat_center,
                                        cutoff_km, btype='low', order=4)
        results['FreqFusion'] = ssv_lp + ship_hp
        print(f"      done {time.time()-t0:.0f}s")
    except Exception as e:
        print(f"      FreqFusion failed: {e}")
        results['FreqFusion'] = None

    print("\n[5/5] Multi-scale evaluation vs shipborne...")
    print("\n  ===== Scale 1: 15'' model vs raw 10s ship points =====")
    print(f"  (含代表性误差: 460m 像素 vs 50m 采样, 重力梯度大处误差大)")
    print(f"  {'Method':<14} {'RMSE':<10} {'RMSE_bc':<10} {'STD':<8} {'Corr':<8} {'Bias':<10} {'N':<10}")
    print("  " + "-" * 72)
    metrics_15s = {}
    for name, grav in results.items():
        if grav is None:
            continue
        m = evaluate_15s_raw(grav, val_points, lon_hr, lat_hr)
        metrics_15s[name] = m
        print(f"  {name:<14} {m['RMSE']:<10.2f} {m['RMSE_bc']:<10.2f} {m['STD']:<8.2f} "
              f"{m['Correlation']:<8.4f} {m['Bias']:<10.2f} {m['N']:<10,}")

    print("\n  ===== Scale 2: 1' model vs 1' ship block-mean (论文标准) =====")
    print(f"  (与 ggad471/NSOAS24/SDUST2022GRA 可直接对比, 典型 5-10 mGal)")
    print(f"  {'Method':<14} {'RMSE':<10} {'RMSE_bc':<10} {'STD':<8} {'Corr':<8} {'Bias':<10} {'N':<10}")
    print("  " + "-" * 72)
    metrics_1m = {}
    for name, grav in results.items():
        if grav is None:
            continue
        m, n = evaluate_1min_grid(grav, val_points, lon_range, lat_range, upscale)
        metrics_1m[name] = m
        print(f"  {name:<14} {m['RMSE']:<10.2f} {m['RMSE_bc']:<10.2f} {m['STD']:<8.2f} "
              f"{m['Correlation']:<8.4f} {m['Bias']:<10.2f} {n:<10,}")

    print("\n  ===== Scale 3: 长波(>160km) / 中波(20-160km) / 短波(<20km) =====")
    print(f"  (去偏差后 RMSE, 反映各波段真实精度)")
    print(f"  {'Method':<14} {'Long RMSE':<12} {'Mid RMSE':<12} {'Short RMSE':<12}")
    print("  " + "-" * 50)
    for name, grav in results.items():
        if grav is None:
            continue
        m_long = evaluate_bandpass(grav, val_points, lon_hr, lat_hr, lon_range, lat_range,
                                    lam_min=160)
        m_mid = evaluate_bandpass(grav, val_points, lon_hr, lat_hr, lon_range, lat_range,
                                   lam_min=20, lam_max=160)
        m_short = evaluate_bandpass(grav, val_points, lon_hr, lat_hr, lon_range, lat_range,
                                     lam_max=20)
        print(f"  {name:<14} {m_long['RMSE_bc']:<12.2f} {m_mid['RMSE_bc']:<12.2f} {m_short['RMSE_bc']:<12.2f}")

    # Reference: shipborne self-precision (~1-2 mGal from crossover analysis)
    print("\n  ===== 参考基准 =====")
    print(f"  船载重力自身精度 (交叉点不符值): ~1-2 mGal (essd-16-4119 报告 2.82 mGal)")
    print(f"  论文 1' 模型 vs 船载: NSOAS24=4.95-6.33, SDUST=4.43-9.69, ggad471=0.6-3 mGal")
    print(f"  本项目 SSV 1' (卫星测高基线): 见 Bilinear 1' 行")
    print(f"  注: RMSE_bc = 去偏差后 RMSE = STD, 反映模型与船载的相对精度")
    print(f"      Bias = 系统偏差 (基准不一致, 可能来自参考椭球/正常重力公式差异)")

    # Save summary
    lines = [
        "=" * 70,
        f"  Independent Cruise Validation v2 (val={VAL_CRUISE})",
        "=" * 70,
        f"  Region: lon=[{lon_range[0]}, {lon_range[1]}], lat=[{lat_range[0]}, {lat_range[1]}]",
        f"  Train: 30 cruises → {n_ship:,} 15'' pixels ({n_ship/ship_grav.size*100:.2f}% coverage)",
        f"  Val:   {VAL_CRUISE}, {len(val_points):,} raw 10s points (fully independent)",
        "",
        "  [Scale 1] 15'' model vs raw 10s ship points (含代表性误差)",
        f"  {'Method':<14} {'RMSE':<10} {'RMSE_bc':<10} {'STD':<8} {'Corr':<8} {'Bias':<10} {'N':<10}",
        "  " + "-" * 72,
    ]
    for name, m in metrics_15s.items():
        lines.append(f"  {name:<14} {m['RMSE']:<10.2f} {m['RMSE_bc']:<10.2f} {m['STD']:<8.2f} "
                     f"{m['Correlation']:<8.4f} {m['Bias']:<10.2f} {m['N']:<10,}")
    lines += [
        "",
        "  [Scale 2] 1' model vs 1' ship block-mean (论文标准)",
        f"  {'Method':<14} {'RMSE':<10} {'RMSE_bc':<10} {'STD':<8} {'Corr':<8} {'Bias':<10} {'N':<10}",
        "  " + "-" * 72,
    ]
    for name, m in metrics_1m.items():
        lines.append(f"  {name:<14} {m['RMSE']:<10.2f} {m['RMSE_bc']:<10.2f} {m['STD']:<8.2f} "
                     f"{m['Correlation']:<8.4f} {m['Bias']:<10.2f} {m.get('N', 0):<10,}")
    lines += [
        "",
        "  [参考] 船载重力自身精度: ~1-2 mGal (crossover)",
        "  [参考] 论文 1' 模型: NSOAS24=4.95-6.33, SDUST=4.43-9.69, ggad471=0.6-3 mGal",
        "  [注] RMSE_bc = 去偏差后 RMSE, 反映模型与船载的相对精度",
        "       Bias = 系统偏差 (基准不一致: 参考椭球/正常重力公式差异)",
    ]
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(OUTPUT_DIR / "independent_validation_v2_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    for name, grav in results.items():
        if grav is not None:
            np.save(OUTPUT_DIR / f"indepval_v2_{name}.npy", grav)
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
    main(
        lon_range=(args.lon_min, args.lon_max),
        lat_range=(args.lat_min, args.lat_max),
        upscale=args.upscale)
