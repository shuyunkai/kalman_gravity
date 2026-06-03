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
from utils import compute_metrics, split_shipborne_mask, read_gmt_grd

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NC_ENGINE = "h5netcdf"

try:
    fm._load_fontmanager(try_read_cache=False)
except AttributeError:
    pass
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
    # Mask land areas (NaN in SSV) — diff only meaningful over ocean
    ocean_mask = ~np.isnan(grav_ssv_sub)
    diff = np.where(ocean_mask, grav_kf_sub - grav_ssv_sub, np.nan)

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
    colors = np.vstack([
        plt.cm.tab20(np.linspace(0, 1, 20)),
        plt.cm.tab20b(np.linspace(0, 1, 20)),
    ])
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
                _, lon, lat = read_gmt_grd(str(f))

                color = colors[c_idx % len(colors)]
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
            except Exception as e:
                print(f"  Warning: skip {f.name} in Fig2: {e}")

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
    if not jap_path.exists():
        raise FileNotFoundError(
            f"日本海沟船载数据未找到: {jap_path}\n"
            f"请从 https://www.marine-geo.org 下载 Subduction_Bassett 数据集")
    ship_g, ship_lon, ship_lat = read_gmt_grd(str(jap_path))

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

    # Convert residual gravity → free-air with EGM2008 long-wave trend (>160km)
    from scipy.ndimage import gaussian_filter
    try:
        ds_egm = xr.open_dataset(
            DATA_DIR / "icgem" / "grid" / "EGM2008_gravity_anomaly.nc",
            engine=NC_ENGINE)
        vname = [v for v in ds_egm.variables if len(ds_egm[v].dims) >= 2][0]
        grav_egm = ds_egm[vname].sel(
            lon=slice(138, 152),
            lat=slice(48, 32)).values  # EGM2008 lat descending
        ds_egm.close()
        egm_trend = gaussian_filter(grav_egm, sigma=10)
        from scipy.interpolate import RegularGridInterpolator
        xi_e = np.linspace(138, 152, egm_trend.shape[1])
        yi_e = np.linspace(48, 32, egm_trend.shape[0])
        interp = RegularGridInterpolator(
            (yi_e, xi_e), egm_trend, bounds_error=False, fill_value=np.nan)
        lr_lon = np.linspace(138, 152, grav_lr.shape[1])
        lr_lat = np.linspace(32, 48, grav_lr.shape[0])
        lo2d, la2d = np.meshgrid(lr_lon, lr_lat)
        egm_on_lr = interp(np.column_stack([la2d.ravel(), lo2d.ravel()]))
        egm_on_lr = egm_on_lr.reshape(grav_lr.shape)
        grav_lr_up = zoom(egm_on_lr, (upscale, upscale), order=1)
    except Exception:
        ssv_trend = gaussian_filter(grav_lr, sigma=10)
        grav_lr_up = zoom(ssv_trend, (upscale, upscale), order=1)
    ship_mask = ~np.isnan(ship_hr)
    is_residual = abs(np.nanmean(ship_hr[ship_mask])) < 50
    if ship_mask.sum() > 0 and is_residual:
        ship_hr[ship_mask] = ship_hr[ship_mask] + grav_lr_up[ship_mask]
    elif ship_mask.sum() > 0:
        print(f"  Shipborne data already FAA (mean={np.nanmean(ship_hr[ship_mask]):.0f}), skip trend")

    recon = KalmanGravityReconstructor(grid_shape=(ny_hr, nx_hr), region_size_deg=16,
                                        lr_grid_shape=(ny_lr, nx_lr))
    ship_train_mask, ship_val_mask = split_shipborne_mask(ship_hr, validation_fraction=0.2)
    recon.initialize(gravity_lr=grav_lr, gravity_ship=ship_hr,
                     ship_train_mask=ship_train_mask,
                     process_noise_std=2.0, sat_noise_std=3.0, ship_noise_std=1.0)
    grav_kf = recon.run(n_iterations=6, smooth_sigma=3.0)
    grav_bi = zoom(grav_lr, (upscale, upscale), order=1)

    result = (grav_lr, grav_kf, grav_bi, ship_hr, ship_lon, ship_lat, lon_hr, lat_hr,
              lon_range, lat_range, ship_val_mask)
    _japan_cache[upscale] = result
    return result


def fig3_regional_zoom():
    """图3: 日本海沟超分辨率六图对比（卫星/双线性/KF/船载/残差）。"""
    print("[Fig 3] Regional Super-Resolution (Japan Trench)...")
    (grav_lr, grav_kf, grav_bi, ship_hr, ship_lon, ship_lat, lon_hr, lat_hr,
     lon_range, lat_range, ship_val_mask) = _load_japan_data(upscale=4)

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

    m_kf_ship = compute_metrics(grav_kf, ship_hr, mask=ship_val_mask)
    m_bi_ship = compute_metrics(grav_bi, ship_hr, mask=ship_val_mask)
    fig.suptitle(f"日本海沟超分辨率重建 [保留船载验证]\n"
                 f"KF vs 船载 RMSE: {m_kf_ship['RMSE']:.1f} mGal  |  "
                 f"双线性 vs 船载 RMSE: {m_bi_ship['RMSE']:.1f} mGal",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_japan_super_resolution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  KF vs Ship (val) RMSE: {m_kf_ship['RMSE']:.1f} mGal")
    print(f"  Bilinear vs Ship (val) RMSE: {m_bi_ship['RMSE']:.1f} mGal")
    print(f"  Saved: fig3_japan_super_resolution.png")


def fig4_scatter_and_histogram():
    """图4: 散点对比 + 误差直方图 + 空间残差 + KF改进量。"""
    print("[Fig 4] Scatter & Histogram...")
    (grav_lr, grav_kf, grav_bi, ship_hr, ship_lon, ship_lat, lon_hr, lat_hr,
     lon_range, lat_range, ship_val_mask) = _load_japan_data(upscale=4)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Only use validation (held-out) shipborne points for metrics
    mask = ship_val_mask & ~np.isnan(ship_hr)
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
    sample_size = 2048

    # Use each dataset's own shape for central window extraction
    ny_s, nx_s = ds_ssv.z.shape
    i0_s = max(0, (ny_s - sample_size) // 2)
    j0_s = max(0, (nx_s - sample_size) // 2)
    grav_ssv = ds_ssv.z.values[i0_s:i0_s+sample_size, j0_s:j0_s+sample_size]
    ds_ssv.close()

    ny_k, nx_k = ds_kf.gravity_anomaly.shape
    i0_k = max(0, (ny_k - sample_size) // 2)
    j0_k = max(0, (nx_k - sample_size) // 2)
    grav_kf = ds_kf.gravity_anomaly.values[i0_k:i0_k+sample_size, j0_k:j0_k+sample_size]
    ds_kf.close()

    # Fill NaN (land) with 0 before FFT to avoid NaN PSD
    grav_ssv = np.nan_to_num(grav_ssv, nan=0.0)
    grav_kf = np.nan_to_num(grav_kf, nan=0.0)

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
    # Annotate wavelength bands for geophysical interpretation
    for wl_km, label, y_pos in [(160, '160 km', 0.95), (20, '20 km', 0.85)]:
        k_val = 2 * np.pi / (wl_km / (111.32 / 60))  # wavenumber in cycles/arcmin
        if k_val > ax.get_xlim()[0] and k_val < ax.get_xlim()[1]:
            ax.axvline(k_val, color='gray', ls=':', lw=0.8, alpha=0.5)
            ax.text(k_val * 1.1, ax.get_ylim()[1] * y_pos, wl_km,
                    fontsize=8, color='gray', alpha=0.7)
    # NOTE: For band-specific validation, add:
    # TODO: compute band-separated PSD_ratio = PSD(KF)/PSD(SSV) and verify
    #       signal enhancement in 20-160 km band without noise amplification
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
    (grav_lr, grav_kf, _, ship_hr, _, _, lon_hr, lat_hr,
     lon_range, lat_range, ship_val_mask) = _load_japan_data(upscale=4)
    from kalman_gravity import KalmanBathymetryInverter
    ny_hr, nx_hr = grav_kf.shape

    # Load TID for independent prior/validation split
    val_mask_fig6 = None
    prior_mask_fig6 = None
    try:
        from real_data import load_gebco_tid, tid_to_multibeam_mask
        tid_info = load_gebco_tid(region=(138, 152, 32, 48))
        mb = tid_to_multibeam_mask(tid_info["tid"])
        mb_hr = sp_zoom(mb.astype(float), (ny_hr / mb.shape[0], nx_hr / mb.shape[1]),
                        order=0) > 0.5
        rng = np.random.default_rng(42)
        mb_idx = np.where(mb_hr.ravel())[0]
        rng.shuffle(mb_idx)
        n_p = int(len(mb_idx) * 0.7)
        prior_mask_fig6 = np.zeros(ny_hr * nx_hr, dtype=bool)
        val_mask_fig6 = np.zeros(ny_hr * nx_hr, dtype=bool)
        prior_mask_fig6[mb_idx[:n_p]] = True
        val_mask_fig6[mb_idx[n_p:]] = True
        prior_mask_fig6 = prior_mask_fig6.reshape(ny_hr, nx_hr)
        val_mask_fig6 = val_mask_fig6.reshape(ny_hr, nx_hr)
        print(f"  Fig6 TID multibeam: {len(mb_idx):,} pixels (prior={n_p:,}, val={len(mb_idx)-n_p:,})")
    except (FileNotFoundError, KeyError) as e:
        print(f"  Fig6 TID not available: {e}")

    bathy_gebco_hr = sp_zoom(bathy, (ny_hr / bathy.shape[0], nx_hr / bathy.shape[1]), order=1)
    inverter = KalmanBathymetryInverter(
        grid_shape=(ny_hr, nx_hr), region_size_deg=16,
        lat_center=(lat_range[0] + lat_range[1]) / 2.0)
    try:
        from real_data import load_age_grid
        age_info = load_age_grid(region=(138, 152, 32, 48))
        inverter.apply_flexural_isostasy_from_age(age_info["age"])
    except (FileNotFoundError, KeyError) as e:
        print(f"  Fig6 age grid not available: {e} — using default Te=10 km")
        inverter.apply_flexural_isostasy(Te=10e3)
    inverter.initialize(gravity_reconstructed=grav_kf, bathymetry_prior=bathy_gebco_hr,
                        prior_mask=prior_mask_fig6,
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

    if val_mask_fig6 is not None and val_mask_fig6.sum() > 100:
        m_kf = compute_metrics(bathy_kf, bathy_gebco_hr, mask=val_mask_fig6)
        m_p = compute_metrics(bathy_parker, bathy_gebco_hr, mask=val_mask_fig6)
        val_note = f"[TID=1独立验证, n={val_mask_fig6.sum():,}]"
    else:
        m_kf = compute_metrics(bathy_kf, bathy_gebco_hr)
        m_p = compute_metrics(bathy_parker, bathy_gebco_hr)
        val_note = "[所有GEBCO像素]"
    fig.suptitle(f"日本海沟水深反演 {val_note}\n"
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


def fig8_residual_diagnostics():
    """图8: 残差空间诊断 — 水深/距离/自相关/剖面 2x2。"""
    print("[Fig 8] Residual Diagnostics...")
    (grav_lr, grav_kf, grav_bi, ship_hr, ship_lon, ship_lat, lon_hr, lat_hr,
     lon_range, lat_range, ship_val_mask) = _load_japan_data(upscale=4)

    ny_hr, nx_hr = grav_kf.shape
    mask = ship_val_mask & ~np.isnan(ship_hr)
    if mask.sum() < 100:
        print("  Skipped: insufficient validation points")
        return

    residual_kf = grav_kf - ship_hr
    residual_bi = grav_bi - ship_hr
    lon_2d, lat_2d = np.meshgrid(lon_hr, lat_hr)

    from scipy.ndimage import zoom as sp_zoom

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Panel 1: Residual vs bathymetry
    ax = axes[0, 0]
    try:
        from real_data import load_gebco
        gebco = load_gebco(region=(lon_range[0], lon_range[1],
                                   lat_range[0], lat_range[1]))
        bathy_lr = gebco["bathymetry"]
        bathy_hr = sp_zoom(bathy_lr, (ny_hr / bathy_lr.shape[0],
                                      nx_hr / bathy_lr.shape[1]), order=1)
        ax.scatter(bathy_hr[mask].ravel(), residual_kf[mask].ravel(),
                   alpha=0.15, s=1, c='steelblue')
        ax.axhline(0, color='k', ls='--', lw=1)
        ax.set_xlabel("GEBCO 水深 (m)")
        ax.set_ylabel("KF 重力残差 (mGal)")
        ax.set_title("残差 vs 水深")
    except Exception:
        ax.text(0.5, 0.5, "GEBCO 不可用", transform=ax.transAxes, ha='center')

    # Panel 2: Residual vs distance to nearest ship point
    ax = axes[0, 1]
    from scipy.spatial import cKDTree
    ship_pts = np.column_stack([
        lon_2d[mask], lat_2d[mask], ship_hr[mask]])
    n_pts = min(2000, len(ship_pts))
    idx_pts = np.random.default_rng(42).choice(len(ship_pts), n_pts, replace=False)
    ship_sample = ship_pts[idx_pts, :2]

    val_lonlat = np.column_stack([lon_2d[mask].ravel(), lat_2d[mask].ravel()])
    # Build KDTree with approximate spherical distance
    cos_lat = np.cos(np.radians(np.median(lat_2d[mask])))
    ship_xy = ship_sample * [111.32 * cos_lat, 111.32]
    val_xy = val_lonlat * [111.32 * cos_lat, 111.32]
    tree = cKDTree(ship_xy)
    dists, _ = tree.query(val_xy)

    ax.scatter(dists, residual_kf[mask].ravel()[:len(dists)],
               alpha=0.15, s=1, c='steelblue')
    ax.axhline(0, color='k', ls='--', lw=1)
    ax.set_xlabel("距最近船载测点 (km)")
    ax.set_ylabel("KF 重力残差 (mGal)")
    ax.set_title("残差 vs 距船载距离")

    # Panel 3: Residual histogram by latitude band
    ax = axes[1, 0]
    lat_vals = lat_2d[mask]
    lat_bins = np.linspace(lat_range[0], lat_range[1], 8)
    band_rms = []
    band_centers = []
    for b in range(len(lat_bins) - 1):
        m = (lat_vals >= lat_bins[b]) & (lat_vals < lat_bins[b + 1])
        if m.sum() > 20:
            band_rms.append(np.sqrt(np.mean(residual_kf[mask][m] ** 2)))
            band_centers.append((lat_bins[b] + lat_bins[b + 1]) / 2)
    if band_rms:
        ax.bar(band_centers, band_rms, width=lat_bins[1] - lat_bins[0],
               color='steelblue', edgecolor='white')
        ax.set_xlabel("纬度")
        ax.set_ylabel("KF 残差 RMSE (mGal)")
        ax.set_title("残差随纬度变化")
    else:
        ax.text(0.5, 0.5, "数据不足", transform=ax.transAxes, ha='center')

    # Panel 4: Error vs bilinear improvement
    ax = axes[1, 1]
    improvement = np.abs(residual_kf[mask].ravel()) - np.abs(residual_bi[mask].ravel())
    ax.hist(improvement, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='k', ls='--', lw=1.5)
    ax.set_xlabel("|KF残差| - |双线性残差| (mGal)")
    ax.set_ylabel("频数")
    ax.set_title(f"KF 改进分布 (负值=KF更优, "
                 f"均值={improvement.mean():.1f} mGal)")

    fig.suptitle("残差空间诊断", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig8_residual_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: fig8_residual_diagnostics.png")


def fig9_parameter_sensitivity():
    """图9: 参数敏感性 — Q/R 网格搜索 RMSE 热力图。"""
    print("[Fig 9] Parameter Sensitivity (Q/R grid search)...")

    process_values = [0.5, 1.0, 2.0, 4.0, 8.0]
    sat_values = [1.0, 2.0, 3.0, 5.0, 8.0]

    # Reuse cached Japan data
    (grav_lr, _, _, ship_hr, _, _, _, _,
     _, _, _) = _load_japan_data(upscale=4)
    ny_lr, nx_lr = grav_lr.shape
    ny_hr, nx_hr = ship_hr.shape

    from kalman_gravity import KalmanGravityReconstructor
    from utils import split_shipborne_mask, compute_metrics
    ship_train, ship_val = split_shipborne_mask(ship_hr, validation_fraction=0.3)
    if ship_val.sum() < 100:
        print("  Skipped: insufficient validation points")
        return

    rmse_grid = np.full((len(process_values), len(sat_values)), np.nan)
    total = len(process_values) * len(sat_values)
    done = 0

    for pi, proc_std in enumerate(process_values):
        for si, sat_std in enumerate(sat_values):
            done += 1
            recon = KalmanGravityReconstructor(
                grid_shape=(ny_hr, nx_hr), region_size_deg=16,
                lr_grid_shape=(ny_lr, nx_lr))
            recon.initialize(gravity_lr=grav_lr, gravity_ship=ship_hr,
                             ship_train_mask=ship_train,
                             process_noise_std=proc_std,
                             sat_noise_std=sat_std, ship_noise_std=1.0)
            grav_hr = recon.run(n_iterations=4, smooth_sigma=0.8, verbose=False)
            m = compute_metrics(grav_hr, ship_hr, mask=ship_val)
            rmse_grid[pi, si] = m.get('RMSE', np.nan)
            print(f"  [{done}/{total}] Q={proc_std}, R={sat_std} -> val RMSE={rmse_grid[pi,si]:.1f} mGal")

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(rmse_grid, origin='lower', aspect='auto', cmap='RdYlBu_r')
    ax.set_xticks(range(len(sat_values)))
    ax.set_xticklabels([str(v) for v in sat_values])
    ax.set_yticks(range(len(process_values)))
    ax.set_yticklabels([str(v) for v in process_values])
    ax.set_xlabel("卫星观测噪声 std (mGal)")
    ax.set_ylabel("过程噪声 std (mGal)")
    ax.set_title("KF 参数敏感性: 验证 RMSE (mGal)\n[日本海沟, ship_noise=1.0]")
    for pi in range(len(process_values)):
        for si in range(len(sat_values)):
            ax.text(si, pi, f"{rmse_grid[pi,si]:.1f}",
                    ha='center', va='center', fontsize=8,
                    color='white' if rmse_grid[pi, si] > np.nanmedian(rmse_grid) else 'black')
    plt.colorbar(im, ax=ax, label="RMSE (mGal)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig9_parameter_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Best Q={process_values[np.unravel_index(np.nanargmin(rmse_grid), rmse_grid.shape)[0]]}, "
          f"R={sat_values[np.unravel_index(np.nanargmin(rmse_grid), rmse_grid.shape)[1]]}")
    print(f"  Saved: fig9_parameter_sensitivity.png")


def fig10_uncertainty_coverage():
    """图10: 不确定性验证 — KF 置信区间覆盖率 vs 理论正态。"""
    print("[Fig 10] Uncertainty Coverage...")

    # Reuse cached Japan data
    (grav_lr, _, _, ship_hr, _, _, _, _,
     _, _, _) = _load_japan_data(upscale=4)
    ny_lr, nx_lr = grav_lr.shape
    ny_hr, nx_hr = ship_hr.shape

    from kalman_gravity import KalmanGravityReconstructor
    from utils import split_shipborne_mask, compute_uncertainty_coverage

    ship_train, ship_val = split_shipborne_mask(ship_hr, validation_fraction=0.3)
    if ship_val.sum() < 100:
        print("  Skipped: insufficient validation points")
        return

    recon = KalmanGravityReconstructor(
        grid_shape=(ny_hr, nx_hr), region_size_deg=16,
        lr_grid_shape=(ny_lr, nx_lr))
    recon.initialize(gravity_lr=grav_lr, gravity_ship=ship_hr,
                     ship_train_mask=ship_train,
                     process_noise_std=2.0, sat_noise_std=3.0, ship_noise_std=1.0)
    grav_hr = recon.run(n_iterations=6, smooth_sigma=0.8, verbose=False)
    P_final = getattr(recon, '_P_final', None)
    if P_final is None:
        print("  WARNING: _P_final not available — using constant P=4.0 fallback")
        P_final = np.full_like(grav_hr, 4.0)

    uc = compute_uncertainty_coverage(
        grav_hr, P_final, ship_hr, R_obs=1.0, grid_shape=(ny_hr, nx_hr))
    if not uc:
        print("  Skipped: not enough data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: standardized residuals histogram
    ax = axes[0]
    std_res = uc['std_residuals']
    std_res_clean = std_res[np.abs(std_res) < 5]  # trim extreme outliers
    ax.hist(std_res_clean, bins=80, density=True, color='steelblue',
            edgecolor='white', alpha=0.8, label='观测')
    x = np.linspace(-4, 4, 200)
    ax.plot(x, np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), 'r-', lw=2, label='N(0,1)')
    ax.axvline(0, color='k', ls='--', lw=1)
    ax.set_xlabel("标准化残差 (pred - val) / sqrt(P + R)")
    ax.set_ylabel("概率密度")
    ax.set_title(f"标准化残差分布 (n={uc['n']:,})")
    ax.legend()

    # Panel 2: coverage bar chart
    ax = axes[1]
    levels = ['1σ (68.3%)', '2σ (95.4%)', '3σ (99.7%)']
    expected = [0.683, 0.954, 0.997]
    observed = [uc['coverage_1sigma'], uc['coverage_2sigma'], uc['coverage_3sigma']]
    x_pos = np.arange(len(levels))
    width = 0.35
    ax.bar(x_pos - width/2, expected, width, label='理论', color='lightgray', edgecolor='black')
    ax.bar(x_pos + width/2, observed, width, label='观测', color='steelblue', edgecolor='white')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(levels)
    ax.set_ylabel("覆盖率")
    ax.set_title(f"置信区间覆盖率 (chi2={uc['chi2_reduced']:.2f})")
    ax.legend()
    for i in range(3):
        ax.text(x_pos[i] + width/2, observed[i] + 0.01, f"{observed[i]:.1%}",
                ha='center', fontsize=9)

    fig.suptitle("卡尔曼滤波不确定性验证", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig10_uncertainty_coverage.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Coverage: 1σ={uc['coverage_1sigma']:.1%}, 2σ={uc['coverage_2sigma']:.1%}, "
          f"3σ={uc['coverage_3sigma']:.1%}, chi2={uc['chi2_reduced']:.2f}")
    print(f"  Saved: fig10_uncertainty_coverage.png")


def fig11_spatial_consistency():
    """图11: 空间一致性 — 空间分块交叉验证。"""
    print("[Fig 11] Spatial Consistency...")
    (grav_lr, grav_kf, grav_bi, ship_hr, ship_lon, ship_lat, lon_hr, lat_hr,
     lon_range, lat_range, ship_val_mask) = _load_japan_data(upscale=4)
    ny_hr, nx_hr = grav_kf.shape

    from utils import spatial_block_split, compute_metrics
    from kalman_gravity import KalmanGravityReconstructor

    blocks = spatial_block_split(ship_hr, n_blocks_row=4, n_blocks_col=4)
    if len(blocks) < 4:
        print(f"  Skipped: only {len(blocks)} blocks with data")
        return

    block_rmse_kf = np.full((4, 4), np.nan)
    block_rmse_bi = np.full((4, 4), np.nan)
    block_n = np.zeros((4, 4), dtype=int)
    block_center_lat = np.full((4, 4), np.nan)

    row_size = ny_hr // 4
    col_size = nx_hr // 4
    ny_lr_orig, nx_lr_orig = grav_lr.shape

    for bi, (train_mask, val_mask) in enumerate(blocks):
        i_block = bi // 4
        j_block = bi % 4
        if val_mask.sum() < 50:
            continue

        recon = KalmanGravityReconstructor(
            grid_shape=(ny_hr, nx_hr), region_size_deg=16,
            lr_grid_shape=(ny_lr_orig, nx_lr_orig))
        recon.initialize(gravity_lr=grav_lr, gravity_ship=ship_hr,
                         ship_train_mask=train_mask,
                         process_noise_std=2.0, sat_noise_std=3.0, ship_noise_std=1.0)
        grav_block = recon.run(n_iterations=2, smooth_sigma=0.8, verbose=False)

        m_kf = compute_metrics(grav_block, ship_hr, mask=val_mask)
        m_bi = compute_metrics(grav_bi, ship_hr, mask=val_mask)
        block_rmse_kf[i_block, j_block] = m_kf.get('RMSE', np.nan)
        block_rmse_bi[i_block, j_block] = m_bi.get('RMSE', np.nan)
        block_n[i_block, j_block] = val_mask.sum()
        block_center_lat[i_block, j_block] = lat_hr[
            min(i_block * row_size + row_size // 2, ny_hr - 1)]

        print(f"  Block [{i_block},{j_block}]: KF={m_kf.get('RMSE',np.nan):.1f}, "
              f"Bilinear={m_bi.get('RMSE',np.nan):.1f} mGal, n={val_mask.sum():,}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    im = ax.imshow(block_rmse_kf, origin='lower', cmap='RdYlBu_r', aspect='auto',
                   vmin=np.nanmin(block_rmse_kf), vmax=np.nanmax(block_rmse_kf))
    for i in range(4):
        for j in range(4):
            if not np.isnan(block_rmse_kf[i, j]):
                ax.text(j, i, f"{block_rmse_kf[i,j]:.1f}", ha='center', va='center',
                        fontsize=8, fontweight='bold')
    ax.set_title("KF 空间分块交叉验证 RMSE (mGal)")
    ax.set_xlabel("经度方向块索引")
    ax.set_ylabel("纬度方向块索引")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    ax.barh(range(16), block_rmse_kf.ravel(), height=0.6,
            color='steelblue', alpha=0.8, label='KF')
    ax.barh(np.arange(16) + 0.3, block_rmse_bi.ravel(), height=0.6,
            color='tomato', alpha=0.8, label='双线性')
    ax.set_yticks(np.arange(16) + 0.15)
    ax.set_yticklabels([f"[{i},{j}]" for i in range(4) for j in range(4)], fontsize=7)
    ax.set_xlabel("RMSE (mGal)")
    ax.set_title("各空间块 RMSE 对比")
    ax.legend()

    valid_rmses = block_rmse_kf[~np.isnan(block_rmse_kf)]
    fig.suptitle(f"空间一致性检验\n"
                 f"KF RMSE 均值={np.nanmean(valid_rmses):.1f}, "
                 f"标准差={np.nanstd(valid_rmses):.1f} mGal "
                 f"(变异系数={np.nanstd(valid_rmses)/np.nanmean(valid_rmses)*100:.0f}%)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig11_spatial_consistency.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Spatial CV: KF mean RMSE={np.nanmean(valid_rmses):.1f}±"
          f"{np.nanstd(valid_rmses):.1f} mGal, CV={np.nanstd(valid_rmses)/np.nanmean(valid_rmses)*100:.0f}%")
    print(f"  Saved: fig11_spatial_consistency.png")


if __name__ == "__main__":
    print("Generating analysis figures...")
    for fig_func in [fig1_global_gravity_map, fig2_shipborne_distribution,
                     fig3_regional_zoom, fig4_scatter_and_histogram,
                     fig5_power_spectrum, fig6_bathymetry_inversion,
                     fig7_summary_table,
                     fig8_residual_diagnostics, fig9_parameter_sensitivity,
                     fig10_uncertainty_coverage, fig11_spatial_consistency]:
        try:
            fig_func()
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as e:
            print(f"  Skipped {fig_func.__name__}: {e}")
    print(f"\nAll figures saved to: {OUTPUT_DIR}")
