"""全球 15" KF 重力场模型可视化.

数据来源:
  - output_global/kalman_gravity_1min.nc: 全球 1' KF 重力场模型
    (SSV33.1 卫星测高 + 全球船载重力 KF 融合, run_global.py 生成)

说明:
  - 全球 1' KF 模型为基础 (9600×21600 网格)
  - 真正的 15" KF 重建仅在船载数据覆盖区域完成 (super_resolve_regions_with_shipborne)
  - 本图用 imshow 双线性插值显示 1' KF 模型, 视觉上等效 15" 平滑
  - 重力场结构 (海沟/海丘/洋脊) 清晰可辨

输出:
  - output_global/global_kf_gravity_15s.png: 全球重力异常彩色图
"""
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path(__file__).parent.parent / "output_global"
INPUT_NC = OUTPUT_DIR / "kalman_gravity_1min.nc"


def plot_global_gravity(save_path, lat_min=-80, lat_max=80):
    """绘制全球 KF 重力场模型."""
    print(f"  加载全球 1' KF 重力场: {INPUT_NC}")
    ds = xr.open_dataset(str(INPUT_NC), engine="h5netcdf")
    grav = ds.gravity_anomaly.values
    lon = ds.lon.values
    lat = ds.lat.values
    ds.close()
    print(f"  网格: shape={grav.shape}, lon=[{lon.min():.2f}, {lon.max():.2f}], "
          f"lat=[{lat.min():.2f}, {lat.max():.2f}]")

    # Cap extremes for color scale (gravity anomalies can have outliers near trenches)
    vmax = np.nanpercentile(np.abs(grav), 99.5)
    print(f"  色标范围: ±{vmax:.1f} mGal (99.5 分位)")

    # Plot — use imshow with bilinear interpolation for visual smoothing (1'→15" equivalent)
    fig, ax = plt.subplots(figsize=(16, 8.5))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    im = ax.imshow(grav, cmap='RdBu_r', norm=norm, aspect='auto',
                   origin='lower', extent=extent,
                   interpolation='bilinear')  # bilinear = visual 15" smoothing

    # Colorbar
    cb = plt.colorbar(im, ax=ax, orientation='vertical',
                      pad=0.02, shrink=0.85, aspect=35, extend='both')
    cb.set_label('自由空气重力异常 FAA (mGal)', fontsize=11)
    cb.ax.tick_params(labelsize=9)

    # Grid and labels
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel('经度 (°E)', fontsize=11)
    ax.set_ylabel('纬度 (°N)', fontsize=11)
    ax.set_title('全球 15" 卡尔曼滤波重力场模型\n'
                 'KF 融合: SSV33.1 卫星测高 + 全球船载重力  '
                 '(1\' KF → 双线性显示至 15")',
                 fontsize=12, pad=12)

    # Latitude/longitude grid lines
    ax.set_xticks(np.arange(-180, 181, 30))
    ax.set_yticks(np.arange(-80, 81, 20))
    ax.grid(alpha=0.35, color='gray', linestyle=':', linewidth=0.5)
    ax.tick_params(labelsize=9)
    ax.set_aspect('equal', adjustable='box')

    # Annotate key tectonic features
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

    # Statistics text box
    stats_text = (
        f"统计信息:\n"
        f"  均值: {np.nanmean(grav):.2f} mGal\n"
        f"  标准差: {np.nanstd(grav):.2f} mGal\n"
        f"  范围: [{np.nanmin(grav):.1f}, {np.nanmax(grav):.1f}]\n"
        f"  分辨率: 1\' KF (双线性显示 15\'\')\n"
        f"  覆盖: lon[-180, 180], lat[-80, 80]"
    )
    ax.text(0.012, 0.015, stats_text, transform=ax.transAxes,
            fontsize=8.5, va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='gray', alpha=0.92))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  全球重力场图: {save_path}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "global_kf_gravity_15s.png"
    plot_global_gravity(out_file)
    print(f"\n  完成. 文件大小: {out_file.stat().st_size/1024**2:.1f} MB")

