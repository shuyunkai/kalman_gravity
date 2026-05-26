"""真实数据加载接口：SSV33.1卫星重力、GEBCO水深、船载数据。"""
import os
import numpy as np
from pathlib import Path
from scipy.interpolate import griddata
import xarray as xr

DATA_DIR = Path(__file__).parent.parent / "data"


def load_ssv_gravity(filepath=None, region=None):
    """加载 SSV 卫星重力 NetCDF 文件，支持按经纬度区域裁剪。"""
    if filepath is None:
        candidates = list(DATA_DIR.glob("**/grav_*.nc"))
        if not candidates:
            raise FileNotFoundError("未找到 SSV 重力文件，请从 https://topex.ucsd.edu/pub/global_grav_1min/ 下载")
        filepath = candidates[0]

    print(f"Loading SSV gravity data from: {filepath}")
    ds = xr.open_dataset(str(filepath), engine="h5netcdf")

    lon_name = "lon" if "lon" in ds.dims else "longitude"
    lat_name = "lat" if "lat" in ds.dims else "latitude"

    if region is not None:
        lon_min, lon_max, lat_min, lat_max = region
        ds_lon = ds[lon_name].values
        if lon_min < 0 and ds_lon.min() >= 0:
            lon_min += 360
            lon_max += 360
        ds = ds.sel(**{lon_name: slice(lon_min, lon_max), lat_name: slice(lat_min, lat_max)})

    gravity = ds["z"].values
    lon = ds[lon_name].values
    lat = ds[lat_name].values
    ds.close()

    print(f"  Loaded gravity: shape={gravity.shape}, lon=[{lon.min():.1f},{lon.max():.1f}], lat=[{lat.min():.1f},{lat.max():.1f}]")
    return {"gravity": gravity, "lon": lon, "lat": lat, "resolution": "1 arc-min"}


def load_ssv_vgg(filepath=None, region=None):
    """加载 SSV 垂直重力梯度 (VGG) NetCDF 文件。"""
    if filepath is None:
        candidates = list(DATA_DIR.glob("**/curv_*.nc"))
        if not candidates:
            raise FileNotFoundError("未找到 SSV curv 文件")
        filepath = candidates[0]

    print(f"Loading SSV VGG data from: {filepath}")
    ds = xr.open_dataset(str(filepath), engine="h5netcdf")

    lon_name = "lon" if "lon" in ds.dims else "longitude"
    lat_name = "lat" if "lat" in ds.dims else "latitude"

    if region is not None:
        lon_min, lon_max, lat_min, lat_max = region
        ds_lon = ds[lon_name].values
        if lon_min < 0 and ds_lon.min() >= 0:
            lon_min += 360
            lon_max += 360
        ds = ds.sel(**{lon_name: slice(lon_min, lon_max), lat_name: slice(lat_min, lat_max)})

    vgg = ds["z"].values
    lon = ds[lon_name].values
    lat = ds[lat_name].values
    ds.close()

    print(f"  Loaded VGG: shape={vgg.shape}")
    return {"vgg": vgg, "lon": lon, "lat": lat, "resolution": "1 arc-min"}


def load_gebco(filepath=None, region=None):
    """加载 GEBCO 全球水深 NetCDF 文件，支持区域裁剪。"""
    if filepath is None:
        candidates = list(DATA_DIR.glob("**/GEBCO*.nc"))
        if not candidates:
            raise FileNotFoundError("GEBCO 文件未找到，请从 https://download.gebco.net 下载")
        filepath = candidates[0]

    print(f"Loading GEBCO bathymetry from: {filepath}")
    ds = xr.open_dataset(str(filepath), engine="h5netcdf")

    lon_name = "lon" if "lon" in ds.dims else "longitude"
    lat_name = "lat" if "lat" in ds.dims else "latitude"
    bathy_name = "elevation" if "elevation" in ds else list(ds.data_vars)[0]

    if region is not None:
        lon_min, lon_max, lat_min, lat_max = region
        ds_lon = ds[lon_name].values
        if lon_min < 0 and ds_lon.min() >= 0:
            lon_min += 360
            lon_max += 360
        ds = ds.sel(**{lon_name: slice(lon_min, lon_max), lat_name: slice(lat_min, lat_max)})

    bathymetry = ds[bathy_name].values
    lon = ds[lon_name].values
    lat = ds[lat_name].values
    ds.close()

    print(f"  Loaded bathymetry: shape={bathymetry.shape}")
    return {"bathymetry": bathymetry, "lon": lon, "lat": lat, "resolution": "15 arc-sec"}


def load_egm2008_gravity(lon_grid, lat_grid, nmax=2190):
    """从 ICGEM 服务或本地球谐系数计算 EGM2008 重力异常。"""
    try:
        from geoclaw.gravity import compute_gravity_anomaly
    except ImportError:
        print("geoclaw 未安装，请使用 ICGEM 在线服务: http://icgem.gfz-potsdam.de/calcgrid")
        return _compute_egm2008_simple(lon_grid, lat_grid, nmax)
    gravity = compute_gravity_anomaly(lon_grid, lat_grid, nmax=nmax)
    return gravity


def _compute_egm2008_simple(lon_grid, lat_grid, nmax=2190):
    """EGM2008 需要球谐系数文件，当前不可用时抛出异常。"""
    raise NotImplementedError(
        "EGM2008 系数文件未安装。请从 http://icgem.gfz-potsdam.de/ 下载。"
        "SSV33.1 可提供等效的海洋重力数据。"
    )


def prepare_real_data_pipeline(region, ssv_file=None, gebco_file=None):
    """为卡尔曼滤波管线准备指定区域的真实数据。"""
    lon_min, lon_max, lat_min, lat_max = region
    ssv_data = load_ssv_gravity(filepath=ssv_file, region=region)
    gravity_lr = ssv_data["gravity"]

    try:
        gebco_data = load_gebco(filepath=gebco_file, region=region)
        bathymetry_prior = gebco_data["bathymetry"]
    except FileNotFoundError:
        print("GEBCO 不可用，水深先验设为 None")
        bathymetry_prior = None

    data = {
        "gravity_lr": gravity_lr, "bathymetry_prior": bathymetry_prior,
        "lon": ssv_data["lon"], "lat": ssv_data["lat"],
        "region": region, "lr_resolution": "1 arc-min",
    }
    return data


def load_shipborne_mgds(cruise_name, data_dir=None):
    """从 MGDS 加载指定航次的船载重力和水深数据。"""
    if data_dir is None:
        data_dir = DATA_DIR / "shipborne"

    cruise_dir = Path(data_dir) / cruise_name
    if not cruise_dir.exists():
        print(f"航次数据未找到: {cruise_dir}")
        print(f"请从 https://www.marine-geo.org 下载，搜索航次: {cruise_name}")
        return None

    grav_files = list(cruise_dir.glob("*.grav")) + list(cruise_dir.glob("*grav*"))
    bathy_files = list(cruise_dir.glob("*.bathy")) + list(cruise_dir.glob("*bathy*"))

    result = {}
    if grav_files:
        result["gravity"] = _parse_mgd77_gravity(grav_files[0])
    if bathy_files:
        result["bathymetry"] = _parse_mgd77_bathymetry(bathy_files[0])
    return result


def _parse_mgd77_gravity(filepath):
    """解析 MGD77 格式重力数据，返回经度、纬度、重力异常。"""
    import pandas as pd
    try:
        df = pd.read_csv(filepath, sep=r"\s+", comment="#")
        return {"lon": df.iloc[:, 0].values, "lat": df.iloc[:, 1].values, "gravity": df.iloc[:, 2].values}
    except Exception:
        return None


def _parse_mgd77_bathymetry(filepath):
    """解析 MGD77 格式水深数据，返回经度、纬度、水深值。"""
    import pandas as pd
    try:
        df = pd.read_csv(filepath, sep=r"\s+", comment="#")
        return {"lon": df.iloc[:, 0].values, "lat": df.iloc[:, 1].values, "bathymetry": df.iloc[:, 2].values}
    except Exception:
        return None


if __name__ == "__main__":
    print("真实数据加载模块")
    print("=" * 50)
    print("使用前请确保数据已下载并放入 data/ 目录。")
    print("运行管线: python src/main.py")
