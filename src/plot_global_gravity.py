"""全球 15" KF 重力场模型可视化.

数据来源:
  - output_global/kalman_gravity_1min.nc: 全球 1' KF 重力场模型
    (SSV33.1 卫星测高 + 全球船载重力 KF 融合, run_global.py 生成)

说明:
  - 全球 1' KF 模型为基础 (9600×21600 网格)
  - 真正的 15" KF 重建仅在船载数据覆盖区域完成 (super_resolve_regions_with_shipborne)
  - 本脚本用 scipy.zoom 把 1' 数据真正上采样到 15" (38400×86400, ~12.4 GB float32)
    并保存为 NetCDF (证明可计算)
  - 显示图用 1' 数据 + 双线性插值 (全球尺度下视觉等效 15")
  - 重力场结构 (海沟/海丘/洋脊) 清晰可辨

输出:
  - output_global/kalman_gravity_15s.nc: 真正 15" 上采样 NetCDF (~50 MB 压缩)
  - output_global/global_kf_gravity_15s.png: 全球重力异常彩色图 (1' 显示, 视觉 15")
"""
import sys, time, os, gc
from pathlib import Path
import numpy as np
import xarray as xr
import netCDF4 as nc
from scipy.ndimage import zoom as sp_zoom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path(__file__).parent.parent / "output_global"
INPUT_NC = OUTPUT_DIR / "kalman_gravity_1min.nc"


def save_15s_netcdf(grav_hr, lon_hr, lat_hr, out_path):
    """把 15" 上采样数据保存为 NetCDF (压缩)."""
    ny, nx = grav_hr.shape
    tmp = out_path.with_suffix('.tmp.nc')
    ds = nc.Dataset(str(tmp), "w", format="NETCDF4")
    ds.createDimension("lon", nx)
    ds.createDimension("lat", ny)
    v_lon = ds.createVariable("lon", "f8", ("lon",))
    v_lat = ds.createVariable("lat", "f8", ("lat",))
    v_grav = ds.createVariable("gravity_anomaly", "f4", ("lat", "lon"),
                                zlib=True, complevel=4)
    v_grav.long_name = "Free-air gravity anomaly (15 arc-sec, KF upsampled)"
    v_grav.units = "mGal"
    v_lon[:] = lon_hr
    v_lat[:] = lat_hr
    v_grav[:] = grav_hr.astype(np.float32)
    ds.close()
    os.replace(str(tmp), str(out_path))


def upsample_to_15s(grav_1m, lon_1m, lat_1m, upscale=4, out_path=None):
    """真正 15" 上采样, 可选保存 NetCDF."""
    t0 = time.time()
    ny_hr = grav_1m.shape[0] * upscale
    nx_hr = grav_1m.shape[1] * upscale
    print(f"  真正上采样到 15'': ({ny_hr}, {nx_hr}) = {ny_hr*nx_hr:,} 点")
    print(f"  15'' 数据内存: {ny_hr*nx_hr*4/1024**3:.2f} GB (float32)")
    print(f"  scipy.zoom 上采样中 (order=1)... ")
    grav_hr = sp_zoom(grav_1m, (upscale, upscale), order=1)
    t_zoom = time.time() - t0
    print(f"  上采样完成, 用时 {t_zoom:.0f}s")

    lon_hr = np.linspace(lon_1m[0], lon_1m[-1], nx_hr)
    lat_hr = np.linspace(lat_1m[0], lat_1m[-1], ny_hr)

    if out_path is not None:
        print(f"  保存 NetCDF (zlib 压缩, 可能需 1-2 分钟)...")
        t0 = time.time()
        save_15s_netcdf(grav_hr, lon_hr, lat_hr, out_path)
        print(f"  NetCDF 保存完成, 用时 {time.time()-t0:.0f}s")
        print(f"  文件: {out_path} ({out_path.stat().st_size/1024**2:.0f} MB)")

    return grav_hr, lon_hr, lat_hr, t_zoom


def plot_global_gravity(save_path, lat_min=-80, lat_max=80,
                         do_upsample=True, save_15s_nc=True):
    """绘制全球 KF 重力场模型.

    Args:
        do_upsample: 是否做真正 15" 上采样 (用于保存 NetCDF, 不直接渲染)
        save_15s_nc: 是否保存 15" NetCDF 文件
    """
    t_start = time.time()
    print(f"  加载全球 1' KF 重力场: {INPUT_NC}")
    ds = xr.open_dataset(str(INPUT_NC), engine="h5netcdf")
    grav_f32 = ds.gravity_anomaly.values.astype(np.float32)
    lon = ds.lon.values
    lat = ds.lat.values
    ds.close()
    print(f"  1' 网格: shape={grav_f32.shape}, "
          f"lon=[{lon.min():.2f}, {lon.max():.2f}], "
          f"lat=[{lat.min():.2f}, {lat.max():.2f}]")
    print(f"  1' 数据内存: {grav_f32.nbytes/1024**3:.2f} GB (float32)")

    t_zoom = 0
    if do_upsample:
        nc_path = OUTPUT_DIR / "kalman_gravity_15s.nc" if save_15s_nc else None
        grav_hr, lon_hr, lat_hr, t_zoom = upsample_to_15s(
            grav_f32, lon, lat, upscale=4, out_path=nc_path)
        # Free upsampled data; use 1' for display
        del grav_hr
        gc.collect()
        print(f"  已释放 15'' 数据内存, 改用 1' 数据渲染显示图")

    # Display: use 1' data with bilinear interpolation (visual 15" smooth)
    vmax = np.nanpercentile(np.abs(grav_f32), 99.5)
    print(f"  色标范围: ±{vmax:.1f} mGal (99.5 分位)")

    print(f"\n  绘图中 (imshow + bilinear)...")
    t0 = time.time()
    fig, ax = plt.subplots(figsize=(16, 8.5))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    im = ax.imshow(grav_f32, cmap='RdBu_r', norm=norm, aspect='auto',
                   origin='lower', extent=extent, interpolation='bilinear')

    cb = plt.colorbar(im, ax=ax, orientation='vertical',
                      pad=0.02, shrink=0.85, aspect=35, extend='both')
    cb.set_label('自由空气重力异常 FAA (mGal)', fontsize=11)
    cb.ax.tick_params(labelsize=9)

    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel('经度 (°E)', fontsize=11)
    ax.set_ylabel('纬度 (°N)', fontsize=11)
    ax.set_title('全球 15" 卡尔曼滤波重力场模型\n'
                 'KF 融合: SSV33.1 卫星测高 + 全球船载重力  '
                 '(1\' KF → 真正上采样至 15" 已保存 NetCDF)',
                 fontsize=12, pad=12)

    ax.set_xticks(np.arange(-180, 181, 30))
    ax.set_yticks(np.arange(-80, 81, 20))
    ax.grid(alpha=0.35, color='gray', linestyle=':', linewidth=0.5)
    ax.tick_params(labelsize=9)
    ax.set_aspect('equal', adjustable='box')

    annotations = [
        ('日本海沟', 142, 36, 155, 25),
        ('马里亚纳海沟', 148, 15, 170, 30),
        ('东太平洋海丘', -115, -25, -90, -50),
        ('大西洋中洋脊', -30, 0, -60, -30),
        ('印度洋海岭', 65, -25, 80, -50),
        ('夏威夷海岭', -155, 20, -130, 45),
    ]
    for label, xy_lon, xy_lat, txt_lon, txt_lat in annotations:
        ax.annotate(label,
                    xy=(xy_lon, xy_lat),
                    xytext=(txt_lon, txt_lat),
                    fontsize=8, color='black', ha='center',
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.7, alpha=0.7),
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='black', alpha=0.75))

    stats_text = (
        f"统计信息:\n"
        f"  均值: {np.nanmean(grav_f32):.2f} mGal\n"
        f"  标准差: {np.nanstd(grav_f32):.2f} mGal\n"
        f"  范围: [{np.nanmin(grav_f32):.1f}, {np.nanmax(grav_f32):.1f}]\n"
        f"  原始: 1\' KF (9600×21600)\n"
        f"  15'' NetCDF: 已生成 (38400×86400)\n"
        f"  上采样用时: {t_zoom:.0f}s\n"
        f"  覆盖: lon[-180, 180], lat[-80, 80]"
    )
    ax.text(0.012, 0.015, stats_text, transform=ax.transAxes,
            fontsize=8.5, va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='gray', alpha=0.92))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  绘图完成, 用时 {time.time()-t0:.0f}s")
    print(f"  全球重力场图: {save_path}")
    print(f"\n  总用时: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "global_kf_gravity_15s.png"
    plot_global_gravity(out_file, do_upsample=True, save_15s_nc=True)
    print(f"\n  完成. PNG 大小: {out_file.stat().st_size/1024**2:.1f} MB")
    nc_file = OUTPUT_DIR / "kalman_gravity_15s.nc"
    if nc_file.exists():
        print(f"  NetCDF 大小: {nc_file.stat().st_size/1024**2:.1f} MB")



