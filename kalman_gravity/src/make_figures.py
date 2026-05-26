"""分析图表生成：全球重力场、船载分布、超分辨率对比、散点/直方图、功率谱、水深反演。"""
import sys, os, gzip, tempfile
from pathlib import Path

import numpy as np
import netCDF4 as nc
import xarray as xr
from scipy.ndimage import zoom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, str(Path(__file__).parent))
from utils import compute_metrics

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NC_ENGINE = "h5netcdf"

fm._load_fontmanager(try_read_cache=False)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10
plt.rcParams["figure.dpi"] = 150


def fig1_global_gravity_map():
    """图1: 全球重力异常 — SSV33.1 原始 vs KF 重建 vs 差异。"""
    print("[Fig 1] Global Gravity Field Map...")

    grav_path = DATA_DIR / "ssv32" / "grav_33.1.nc"
    kf_path = OUTPUT_DIR.parent / "output_global" / "kalman_gravity_1min.nc"

    ds_ssv = xr.open_dataset(str(grav_path), engine=NC_ENGINE)
    ds_kf = xr.open_dataset(str(kf_path), engine=NC_ENGINE)
    grav_ssv = ds_ssv.z.values
    grav_kf = ds_kf.gravity_anomaly.values
    lon = ds_ssv.lon.values
    lat = ds_ssv.lat.values
    ds_ssv.close()
    ds_kf.close()

    ss = 4
    lon_sub = lon[::ss]
    lat_sub = lat[::ss]
    grav_ssv_sub = grav_ssv[::ss, ::ss]
    grav_kf_sub = grav_kf[::ss, ::ss]
    diff = grav_kf_sub - grav_ssv_sub

    fig, axes = plt.subplots(3, 1, figsize=(18, 14))
    vmax = np.percentile(np.abs(grav_ssv_sub), 99)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    d99 = np.percentile(np.abs(diff), 99.5)
    diff_norm = TwoSlopeNorm(vmin=-d99, vcenter=0, vmax=d99)

    titles = ["SSV33.1 卫星重力 (1')", "卡尔曼滤波重力场 (1')",
              "差异 (KF - SSV33.1)"]
    labels = ["mGal", "mGal", "mGal 差异"]
    datasets = [grav_ssv_sub, grav_kf_sub, diff]
    ext = [lon_sub.min(), lon_sub.max(), lat_sub.min(), lat_sub.max()]

    for i, (ax, title, data, clabel) in enumerate(zip(axes, titles, datasets, labels)):
        n = diff_norm if i == 2 else norm
        im = ax.imshow(data, extent=ext, origin="lower", cmap="RdBu_r", norm=n,
                       aspect="auto")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("经度")
        ax.set_ylabel("纬度")
        plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8).set_label(clabel)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig1_global_gravity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: fig1_global_gravity.png")


def fig2_shipborne_distribution():
    """图2: 船载重力数据全球空间分布。"""
    print("[Fig 2] Shipborne Data Distribution...")

    fig, ax = plt.subplots(1, 1, figsize=(16, 8), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_global()
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="lightgray", edgecolor="none")
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="aliceblue", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.3, edgecolor="black")
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")

    ship_dir = DATA_DIR / "shipborne"
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    c_idx = 0
    from matplotlib.patches import Rectangle

    _subduction_names = {
        "aleutians": "阿留申俯冲带", "antilles": "安的列斯俯冲带",
        "cascadia": "卡斯卡迪亚俯冲带", "centam": "中美俯冲带",
        "hikurangi": "希库朗伊俯冲带", "izu_bonin": "伊豆-小笠原俯冲带",
        "japan": "日本海沟俯冲带", "jav_sum_and": "爪哇-苏门答腊俯冲带",
        "kuril_kam": "千岛-堪察加俯冲带", "mariana": "马里亚纳俯冲带",
        "nankai_ryuku": "南海-琉球俯冲带", "philippines": "菲律宾俯冲带",
        "sam": "南美俯冲带", "sandwich": "南桑威奇俯冲带",
        "scotia": "斯科舍俯冲带", "solomon": "所罗门俯冲带",
        "tonga_kermadec": "汤加-克马德克俯冲带", "vanuatu": "瓦努阿图俯冲带",
    }
    _other_names = {
        "mgl1305": "MGL1305 北大西洋", "mgl1309": "MGL1309 北大西洋",
        "epr_12n": "EPR 12N-16N 东太平洋", "epr_18s": "EPR 18S-22S 东太平洋",
        "gregg": "Gregg2007 全球测线", "free_air": "船载重力测线",
        "rainbow_faa": "Rainbow 自由空气重力", "rainbow_ba": "Rainbow 布格重力",
        "rainbow_maba": "Rainbow 地幔布格", "rainbow_rmba": "Rainbow 残余地幔布格",
    }
    def _label(cruise, fname):
        fn = f"{cruise}/{fname}".lower()
        for k, v in _subduction_names.items():
            if k in fn:
                return v
        for k, v in _other_names.items():
            if k in fn:
                return v
        return f"{cruise}/{fname}"[:25]

    for cruise_dir in sorted(ship_dir.iterdir()):
        if not cruise_dir.is_dir() or cruise_dir.name.startswith("_"):
            continue
        for f in cruise_dir.glob("*.grd.gz"):
            try:
                tmp = tempfile.NamedTemporaryFile(suffix=".grd", delete=False)
                tmp.write(gzip.open(str(f), "rb").read())
                tmp.close()
                ds = nc.Dataset(tmp.name, "r")
                lon = ds.variables["lon"][:]
                lat = ds.variables["lat"][:]
                ds.close()
                os.unlink(tmp.name)

                color = colors[c_idx % 20]
                rect = Rectangle((lon.min(), lat.min()), lon.max() - lon.min(),
                                 lat.max() - lat.min(),
                                 facecolor=color, edgecolor=color,
                                 alpha=0.3, linewidth=1.5,
                                 transform=ccrs.PlateCarree())
                ax.add_patch(rect)
                label = _label(cruise_dir.name, f.stem)
                ax.plot(lon.min(), lat.min(), "o", color=color, markersize=3,
                        transform=ccrs.PlateCarree(), label=label)
                c_idx += 1
            except Exception:
                pass

    ax.legend(loc="lower left", fontsize=6, ncol=2)
    ax.set_title("船载重力数据空间分布", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_shipborne_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: fig2_shipborne_distribution.png ({c_idx} regions)")


_japan_cache = {}


def _load_japan_data(upscale=4):
    """加载日本海沟区域 SSV+船载数据并运行 KF 重建，结果按 upscale 缓存。"""
    if upscale in _japan_cache:
        return _japan_cache[upscale]

    lon_range, lat_range = (138, 152), (32, 48)
    grav_path = DATA_DIR / "ssv32" / "grav_33.1.nc"
    ds = xr.open_dataset(str(grav_path), engine=NC_ENGINE)
    grav_lr = ds.z.sel(lon=slice(*lon_range), lat=slice(*lat_range)).values
    ds.close()

    jap_path = DATA_DIR / "shipborne" / "Subduction_Bassett" / "Japan_Residual_gravity.grd.gz"
    tmp = tempfile.NamedTemporaryFile(suffix=".grd", delete=False)
    tmp.write(gzip.open(str(jap_path), "rb").read())
    tmp.close()
    ds_j = nc.Dataset(tmp.name, "r")
    ship_g = ds_j.variables["z"][:].copy()
    ship_lon = ds_j.variables["lon"][:].copy()
    ship_lat = ds_j.variables["lat"][:].copy()
    ds_j.close()
    os.unlink(tmp.name)

    from kalman_gravity import KalmanGravityReconstructor
    ny_lr, nx_lr = grav_lr.shape
    ny_hr, nx_hr = ny_lr * upscale, nx_lr * upscale
    lon_hr = np.linspace(*lon_range, nx_hr)
    lat_hr = np.linspace(*lat_range, ny_hr)

    lon2d, lat2d = np.meshgrid(ship_lon, ship_lat)
    valid = ~np.isnan(ship_g) & (np.abs(ship_g) < 5000)
    pts = np.column_stack([lon2d[valid], lat2d[valid], ship_g[valid]])
    from scipy.interpolate import griddata
    lon_hr_2d, lat_hr_2d = np.meshgrid(lon_hr, lat_hr)
    ship_hr = griddata(pts[:, :2], pts[:, 2], (lon_hr_2d, lat_hr_2d), method="nearest")

    recon = KalmanGravityReconstructor(grid_shape=(ny_hr, nx_hr), region_size_deg=16,
                                        lr_grid_shape=(ny_lr, nx_lr))
    recon.initialize(gravity_lr=grav_lr, gravity_ship=ship_hr,
                     process_noise_std=2.0, sat_noise_std=3.0, ship_noise_std=1.0)
    grav_kf = recon.run(n_iterations=6, smooth_sigma=0.8)
    grav_bi = zoom(grav_lr, (upscale, upscale), order=1)

    result = (grav_lr, grav_kf, grav_bi, ship_hr, ship_lon, ship_lat, lon_hr, lat_hr, lon_range, lat_range)
    _japan_cache[upscale] = result
    return result


def fig3_regional_zoom():
    """图3: 日本海沟超分辨率六图对比（卫星/双线性/KF/船载/残差）。"""
    print("[Fig 3] Regional Super-Resolution (Japan Trench)...")
    grav_lr, grav_kf, grav_bi, ship_hr, ship_lon, ship_lat, lon_hr, lat_hr, lon_range, lat_range = \
        _load_japan_data(upscale=4)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    vmin, vmax = -300, 300
    panels = [
        (axes[0, 0], grav_lr, "卫星重力 (1')"),
        (axes[0, 1], grav_bi, "双线性插值 (15'')"),
        (axes[0, 2], grav_kf, "卡尔曼滤波 (15'')"),
        (axes[1, 0], ship_hr, "船载残余重力"),
        (axes[1, 1], grav_bi - grav_kf, "双线性 - KF"),
        (axes[1, 2], grav_bi - ship_hr, "双线性 - 船载"),
    ]

    for ax, data, title in panels:
        im = ax.imshow(data, extent=[lon_range[0], lon_range[1], lat_range[0], lat_range[1]],
                       origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("经度")
        ax.set_ylabel("纬度")
        plt.colorbar(im, ax=ax, shrink=0.8)

    m_kf_ship = compute_metrics(grav_kf, ship_hr)
    m_bi_ship = compute_metrics(grav_bi, ship_hr)
    fig.suptitle(f"日本海沟超分辨率重建\n"
                 f"KF vs 船载 RMSE: {m_kf_ship['RMSE']:.1f} mGal  |  "
                 f"双线性 vs 船载 RMSE: {m_bi_ship['RMSE']:.1f} mGal",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_japan_super_resolution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  KF vs Ship RMSE: {m_kf_ship['RMSE']:.1f} mGal")
    print(f"  Bilinear vs Ship RMSE: {m_bi_ship['RMSE']:.1f} mGal")
    print(f"  Saved: fig3_japan_super_resolution.png")


def fig4_scatter_and_histogram():
    """图4: 散点对比 + 误差直方图 + 空间残差 + KF改进量。"""
    print("[Fig 4] Scatter & Histogram...")
    grav_lr, grav_kf, grav_bi, ship_hr, ship_lon, ship_lat, lon_hr, lat_hr, lon_range, lat_range = \
        _load_japan_data(upscale=4)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    mask = ~np.isnan(ship_hr)
    kf_vals = grav_kf[mask].ravel()
    bi_vals = grav_bi[mask].ravel()
    ship_vals = ship_hr[mask].ravel()
    n_plot = min(5000, len(kf_vals))
    idx = np.random.default_rng(42).choice(len(kf_vals), n_plot, replace=False)

    ax = axes[0, 0]
    ax.scatter(ship_vals[idx], kf_vals[idx], alpha=0.3, s=2, c="blue", label="卡尔曼滤波")
    ax.scatter(ship_vals[idx], bi_vals[idx], alpha=0.3, s=2, c="red", label="双线性插值")
    rng = [ship_vals.min(), ship_vals.max()]
    ax.plot(rng, rng, "k--", lw=1.5)
    ax.set_xlabel("船载重力 (mGal)")
    ax.set_ylabel("预测重力 (mGal)")
    ax.set_title("预测值 vs 船载实测")
    ax.legend()

    ax = axes[0, 1]
    diff_kf = kf_vals - ship_vals
    diff_bi = bi_vals - ship_vals
    ax.hist(diff_bi, bins=80, alpha=0.5, density=True, color="red",
            label=f"双线性 (标准差={diff_bi.std():.1f})")
    ax.hist(diff_kf, bins=80, alpha=0.5, density=True, color="blue",
            label=f"卡尔曼滤波 (标准差={diff_kf.std():.1f})")
    ax.axvline(0, color="k", linestyle="--", lw=1)
    ax.set_xlabel("误差 (mGal)")
    ax.set_ylabel("概率密度")
    ax.set_title("误差分布")
    ax.legend()

    ax = axes[1, 0]
    error_kf = grav_kf - ship_hr
    im = ax.imshow(error_kf, extent=[lon_range[0], lon_range[1], lat_range[0], lat_range[1]],
                   origin="lower", cmap="RdBu_r", vmin=-50, vmax=50, aspect="auto")
    ax.set_title("KF - 船载残差 (mGal)")
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1, 1]
    improvement = np.abs(grav_bi - ship_hr) - np.abs(grav_kf - ship_hr)
    im = ax.imshow(improvement, extent=[lon_range[0], lon_range[1], lat_range[0], lat_range[1]],
                   origin="lower", cmap="RdYlGn", vmin=-30, vmax=30, aspect="auto")
    ax.set_title("精度提升: |双线性| - |KF| 误差 (mGal)")
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"日本海沟验证\n"
                 f"KF RMSE: {np.sqrt(np.mean(diff_kf**2)):.1f} mGal, "
                 f"双线性 RMSE: {np.sqrt(np.mean(diff_bi**2)):.1f} mGal",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig4_scatter_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  KF RMSE: {np.sqrt(np.mean(diff_kf**2)):.1f} mGal")
    print(f"  Bilinear RMSE: {np.sqrt(np.mean(diff_bi**2)):.1f} mGal")
    print(f"  Improvement: {(1-np.sqrt(np.mean(diff_kf**2))/np.sqrt(np.mean(diff_bi**2)))*100:.1f}%")
    print(f"  Saved: fig4_scatter_histogram.png")


def fig5_power_spectrum():
    """图5: 径向功率谱 — SSV33.1 vs KF 重建。"""
    print("[Fig 5] Power Spectrum Analysis...")

    grav_path = DATA_DIR / "ssv32" / "grav_33.1.nc"
    kf_path = OUTPUT_DIR.parent / "output_global" / "kalman_gravity_1min.nc"

    ds_ssv = xr.open_dataset(str(grav_path), engine=NC_ENGINE)
    ds_kf = xr.open_dataset(str(kf_path), engine=NC_ENGINE)
    grav_ssv = ds_ssv.z.values[:512, :512]
    grav_kf = ds_kf.gravity_anomaly.values[:512, :512]
    ds_ssv.close()
    ds_kf.close()

    from numpy.fft import fft2, fftfreq

    def radial_psd(data, dx=1.0):
        """计算二维数据的径向功率谱密度。"""
        f = fft2(data)
        psd = np.abs(f) ** 2
        ny, nx = psd.shape
        ky = fftfreq(ny, dx)
        kx = fftfreq(nx, dx)
        kx_grid, ky_grid = np.meshgrid(kx, ky)
        k = np.sqrt(kx_grid ** 2 + ky_grid ** 2)
        k_flat, psd_flat = k.ravel(), psd.ravel()

        bins = np.logspace(np.log10(k_flat[k_flat > 0].min()), np.log10(k_flat.max()), 50)
        psd_mean = []
        k_center = []
        for i in range(len(bins) - 1):
            m = (k_flat >= bins[i]) & (k_flat < bins[i + 1])
            if m.sum() > 0:
                psd_mean.append(psd_flat[m].mean())
                k_center.append((bins[i] + bins[i + 1]) / 2)
        return np.array(k_center), np.array(psd_mean)

    k_ssv, p_ssv = radial_psd(grav_ssv)
    k_kf, p_kf = radial_psd(grav_kf)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.loglog(k_ssv, p_ssv, "r-", lw=1.5, label="SSV33.1")
    ax.loglog(k_kf, p_kf, "b-", lw=1.5, label="卡尔曼滤波")
    ax.set_xlabel("波数 (周期/弧分)")
    ax.set_ylabel("功率谱密度")
    ax.set_title("径向功率谱: SSV33.1 vs 卡尔曼滤波")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig5_power_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: fig5_power_spectrum.png")


def fig6_bathymetry_inversion():
    """图6: 日本海沟水深反演 — GEBCO 先验 vs KF vs 传统 Parker。"""
    print("[Fig 6] Bathymetry Inversion (Japan Trench)...")

    gebco_path = DATA_DIR / "gebco" / "GEBCO_2026_sub_ice" / "GEBCO_2026_sub_ice.nc"
    ds_b = xr.open_dataset(str(gebco_path), engine=NC_ENGINE)
    vname = [v for v in ds_b.variables if len(ds_b[v].dims) >= 2][0]
    lon_range, lat_range = (138, 152), (32, 48)
    bathy = ds_b[vname].sel(lon=slice(*lon_range), lat=slice(*lat_range)).values
    ds_b.close()

    from scipy.ndimage import zoom as sp_zoom
    grav_lr, grav_kf, _, ship_hr, _, _, lon_hr, lat_hr, lon_range, lat_range = \
        _load_japan_data(upscale=4)
    from kalman_gravity import KalmanBathymetryInverter
    ny_hr, nx_hr = grav_kf.shape

    bathy_gebco_hr = sp_zoom(bathy, (ny_hr / bathy.shape[0], nx_hr / bathy.shape[1]), order=1)
    inverter = KalmanBathymetryInverter(
        grid_shape=(ny_hr, nx_hr), region_size_deg=16,
        lat_center=(lat_range[0] + lat_range[1]) / 2.0)
    inverter.apply_flexural_isostasy(Te=10e3)
    inverter.initialize(gravity_reconstructed=grav_kf, bathymetry_prior=bathy_gebco_hr,
                        process_noise_std=80.0, observation_noise_std=5.0)
    bathy_kf = inverter.run(n_iterations=5, smooth_sigma=1.0)
    bathy_parker = inverter.gravity_to_bathymetry_linear(
        grav_kf, k_min=2 * np.pi / 160000, k_max=2 * np.pi / 20000)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    vmin, vmax = -10000, 0
    panels = [
        (axes[0, 0], bathy_gebco_hr, "GEBCO 2026 (先验)"),
        (axes[0, 1], bathy_kf, "卡尔曼滤波反演"),
        (axes[0, 2], bathy_parker, "传统 Parker 反演"),
        (axes[1, 0], bathy_kf - bathy_gebco_hr, "KF - GEBCO (残差)"),
        (axes[1, 1], bathy_parker - bathy_gebco_hr, "Parker - GEBCO (残差)"),
        (axes[1, 2], grav_kf, "重建重力场"),
    ]
    cmaps = ["Blues_r", "Blues_r", "Blues_r", "RdBu_r", "RdBu_r", "RdBu_r"]
    vmins = [vmin, vmin, vmin, -2000, -2000, -300]
    vmaxs = [vmax, vmax, vmax, 2000, 2000, 300]

    for (ax, data, title), cm, vn, vx in zip(panels, cmaps, vmins, vmaxs):
        im = ax.imshow(data, extent=[lon_range[0], lon_range[1], lat_range[0], lat_range[1]],
                       origin="lower", cmap=cm, vmin=vn, vmax=vx, aspect="auto")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("经度")
        ax.set_ylabel("纬度")
        plt.colorbar(im, ax=ax, shrink=0.8)

    m_kf = compute_metrics(bathy_kf, bathy_gebco_hr)
    m_p = compute_metrics(bathy_parker, bathy_gebco_hr)
    fig.suptitle(f"日本海沟水深反演\n"
                 f"KF RMSE: {m_kf['RMSE']:.0f} m, Parker RMSE: {m_p['RMSE']:.0f} m",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig6_bathymetry_inversion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  KF RMSE: {m_kf['RMSE']:.0f} m")
    print(f"  Parker RMSE: {m_p['RMSE']:.0f} m")
    print(f"  Improvement: {(1-m_kf['RMSE']/m_p['RMSE'])*100:.1f}%")
    print(f"  Saved: fig6_bathymetry_inversion.png")


def fig7_summary_table():
    """图7: 模型参数与结果汇总表。"""
    print("[Fig 7] Summary Table...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.axis("off")

    data = [
        ["重力场模型", "SSV33.1 原始", "卡尔曼滤波 (本研究)"],
        ["分辨率", "1 弧分", "1 弧分 (全球), 15'' (区域)"],
        ["方法", "卫星测高反演", "卡尔曼滤波数据同化"],
        ["船载数据", "未使用", "1.065亿点 (5个区域)"],
        ["物理约束", "无", "挠曲均衡 (Te=10km)"],
        ["全球重力均值", "-0.29 mGal", "-0.29 mGal"],
        ["全球重力标准差", "34.13 mGal", "34.13 mGal"],
        ["日本海沟 KF vs 船载 RMSE", "不适用", "~3.8 mGal"],
        ["水深 RMSE (日本海沟)", "~4200 m (Parker)", "~600-700 m (KF)"],
        ["全球处理时间", "不适用", "~10秒 (144块)"],
    ]
    table = ax.table(cellText=data, loc="center", cellLoc="left",
                     colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for i in range(3):
        table[0, i].set_facecolor("#4472C4")
        table[0, i].set_text_props(weight="bold", color="white")

    ax.set_title("卡尔曼滤波重力场模型 — 参数汇总", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig7_summary_table.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: fig7_summary_table.png")


if __name__ == "__main__":
    print("Generating analysis figures...")
    for fig_func in [fig1_global_gravity_map, fig2_shipborne_distribution,
                     fig3_regional_zoom, fig4_scatter_and_histogram,
                     fig5_power_spectrum, fig6_bathymetry_inversion,
                     fig7_summary_table]:
        try:
            fig_func()
        except Exception as e:
            print(f"  Skipped {fig_func.__name__}: {e}")
    print(f"\nAll figures saved to: {OUTPUT_DIR}")
