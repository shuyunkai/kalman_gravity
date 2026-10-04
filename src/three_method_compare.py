"""3 方法对比 — 双线性 / 频域融合 / KF (地形先验).

区域: lon=132-152°E, lat=26-42°N (20°×16°)
- 1' 网格: 1200×960
- 15'' 网格: 4800×3840 = 18.4M 像素
- 验证: MR22-04 独立航次 (完全未参与训练)

3 方法:
  1. 双线性插值 (SSV 1' 上采样)
  2. 频域融合 (Butterworth 滤波, SSV 低频 + 船载高频)
  3. KF (卡尔曼滤波, SSV 块常数 + GEBCO 布格高频先验)
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
from freq_fusion_baseline import freq_fusion_reconstruct
from kf_novel import BathyPriorKF

OUTPUT_DIR = Path(__file__).parent.parent / "output_real"

METHOD_BILINEAR = '双线性插值'
METHOD_FREQFUSION = '频域融合'
METHOD_KF = 'KF'


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


def load_region_data(lon_range, lat_range, val_cruise=VAL_CRUISE, upscale=4):
    """加载区域数据: SSV 1' + 船载 (按 val_cruise 切分) + GEBCO 15'' 水深."""
    gravity_lr, _, lon_lr, lat_lr = load_ssv_region(lon_range, lat_range, load_vgg=False)
    ny_lr, nx_lr = gravity_lr.shape
    ny_hr, nx_hr = ny_lr * upscale, nx_lr * upscale
    lon_hr = np.linspace(lon_range[0], lon_range[1], nx_hr)
    lat_hr = np.linspace(lat_range[0], lat_range[1], ny_hr)
    print(f"  SSV 1': {gravity_lr.shape}  目标 15'': ({ny_hr}, {nx_hr}) "
          f"({ny_hr*nx_hr/1e6:.1f}M 像素)")

    train_points, val_points = load_all_cruises_split(lon_range, lat_range, val_cruise)
    print(f"  训练: {len(train_points):,} 点  验证 {val_cruise}: {len(val_points):,} 点")

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

    return {
        'gravity_lr': gravity_lr, 'lon_lr': lon_lr, 'lat_lr': lat_lr,
        'lon_hr': lon_hr, 'lat_hr': lat_hr,
        'ny_lr': ny_lr, 'nx_lr': nx_lr, 'ny_hr': ny_hr, 'nx_hr': nx_hr,
        'ship_grav': ship_grav, 'train_mask': train_mask,
        'bathy_15s': bathy_15s, 'ssv_hr': ssv_hr,
        'train_points': train_points, 'val_points': val_points,
    }


def build_three_methods(data, alpha=0.2, n_iter=6, smooth_sigma=15.0):
    """构建 3 方法: 双线性 / 频域融合 / KF."""
    ssv_hr = data['ssv_hr']
    gravity_lr = data['gravity_lr']
    ship_grav = data['ship_grav']
    train_mask = data['train_mask']
    bathy_15s = data['bathy_15s']
    lon_hr, lat_hr = data['lon_hr'], data['lat_hr']
    lon_range, lat_range = data.get('lon_range'), data.get('lat_range')
    ny_hr, nx_hr = data['ny_hr'], data['nx_hr']
    ny_lr, nx_lr = data['ny_lr'], data['nx_lr']
    region_size = lat_range[1] - lat_range[0]

    models = {}
    times = {}

    t0 = time.time()
    models[METHOD_BILINEAR] = ssv_hr.copy()
    times[METHOD_BILINEAR] = time.time() - t0
    print(f"  双线性: {times[METHOD_BILINEAR]:.1f}s")

    t0 = time.time()
    print("  频域融合...")
    models[METHOD_FREQFUSION] = freq_fusion_reconstruct(
        ssv_hr=ssv_hr, ship_grav=ship_grav, train_mask=train_mask,
        lon_hr=lon_hr, lat_hr=lat_hr,
        lon_range=lon_range, lat_range=lat_range)
    times[METHOD_FREQFUSION] = time.time() - t0
    print(f"  频域融合: {times[METHOD_FREQFUSION]:.1f}s")

    t0 = time.time()
    print(f"  KF (α={alpha})...")
    kf = BathyPriorKF(
        hr_grid_shape=(ny_hr, nx_hr), lr_grid_shape=(ny_lr, nx_lr),
        region_size_deg=region_size, alpha=alpha)
    models[METHOD_KF] = kf.build(
        gravity_lr=gravity_lr, ship_15s=ship_grav, train_mask=train_mask,
        lon_hr=lon_hr, lat_hr=lat_hr, lon_range=lon_range, lat_range=lat_range,
        upscale=4, bathy_15s=bathy_15s,
        n_iter=n_iter, smooth_sigma=smooth_sigma)
    times[METHOD_KF] = time.time() - t0
    print(f"  KF: {times[METHOD_KF]:.1f}s")

    return models, times
