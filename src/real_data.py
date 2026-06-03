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


def load_gebco_tid(filepath=None, region=None):
    """加载 GEBCO TID (Type Identifier) 类型标识格网。

    TID 值含义 (GEBCO 编码):
      0     = 陆地
      10-17 = 实测多波束/单波束 → 独立测量，可用于约束
      40-48 = 卫星重力反演填充 → 循环推导，绝对不能用于约束
      70-72 = 插值/预生成格网 → 循环推导

    Returns dict with tid, lon, lat.
    """
    if filepath is None:
        candidates = list(DATA_DIR.glob("**/GEBCO*TID*.nc")) + \
                      list(DATA_DIR.glob("**/gebco*/*TID*.nc"))
        if not candidates:
            raise FileNotFoundError(
                "GEBCO TID 文件未找到，请从 https://download.gebco.net 下载")
        filepath = candidates[0]

    print(f"Loading GEBCO TID from: {filepath}")
    # TID file is large (3.5 GB) — try netcdf4 engine first, fall back to h5netcdf
    try:
        ds = xr.open_dataset(str(filepath), engine="netcdf4")
    except Exception:
        print("  netcdf4 engine failed, trying h5netcdf...")
        ds = xr.open_dataset(str(filepath), engine="h5netcdf")

    lon_name = "lon" if "lon" in ds.dims else "longitude"
    lat_name = "lat" if "lat" in ds.dims else "latitude"

    # GEBCO TID 使用 -180/180 经度，需转换
    if region is not None:
        lon_min, lon_max, lat_min, lat_max = region
        ds_lon = ds[lon_name].values
        if lon_min < 0 and ds_lon.min() >= 0:
            lon_min += 360
            lon_max += 360
        ds = ds.sel(**{lon_name: slice(lon_min, lon_max),
                       lat_name: slice(lat_min, lat_max)})

    # TID 变量名可能是 'tid' 或第一个 2D data variable
    tid_name = "tid" if "tid" in ds else [v for v in ds.data_vars
                                           if len(ds[v].dims) >= 2][0]
    tid = ds[tid_name].values.astype(np.int16)
    lon_t = ds[lon_name].values
    lat_t = ds[lat_name].values
    ds.close()

    n_measured = ((tid >= 10) & (tid < 20)).sum()
    n_inverted = ((tid >= 40) & (tid < 50)).sum()
    print(f"  TID: shape={tid.shape}, measured(TID10-17)={n_measured:,}, "
          f"inverted(TID40-48)={n_inverted:,}")
    return {"tid": tid, "lon": lon_t, "lat": lat_t}


def tid_to_multibeam_mask(tid_data):
    """将 TID 格网转换为布尔掩码: 实测多波束区域为 True。

    GEBCO TID 编码:
      0     = 陆地
      10-17 = 实测 (多波束/单波束/等深线/Olex/数字海图等)
      40-48 = 卫星重力反演填充 (GRIDDED/SID/深度预测)
      70-72 = 预生成/插值格网
    """
    return (tid_data >= 10) & (tid_data < 20)


def load_age_grid(filepath=None, region=None):
    """加载 Seton et al. (2020) 现代洋壳年龄格网。

    分辨率: 2 弧分 (5401x10801)
    值: 0-340 Ma, 0=陆地/不确定

    Returns dict with age (Ma), lon, lat.
    """
    if filepath is None:
        candidates = (list(DATA_DIR.glob("**/Grids/**/age.*GeeK2007*2m*.nc")) +
                      list(DATA_DIR.glob("**/Grids/**/age.*GTS2012*2m*.nc")))
        if not candidates:
            raise FileNotFoundError(
                "洋壳年龄格网未找到。请从 https://zenodo.org/records/5341342 "
                "下载 Grids.zip, 解压后放入 data/Grids/")
        filepath = candidates[0]

    print(f"Loading ocean crust age from: {filepath}")
    ds = xr.open_dataset(str(filepath), engine="h5netcdf")

    lon_name = "lon" if "lon" in ds.dims else "longitude"
    lat_name = "lat" if "lat" in ds.dims else "latitude"
    age_var = [v for v in ds.data_vars if len(ds[v].dims) >= 2][0]

    if region is not None:
        lon_min, lon_max, lat_min, lat_max = region
        ds_lon = ds[lon_name].values
        if lon_min < 0 and ds_lon.min() >= 0:
            lon_min += 360
            lon_max += 360
        ds = ds.sel(**{lon_name: slice(lon_min, lon_max),
                       lat_name: slice(lat_min, lat_max)})

    age = ds[age_var].values.astype(np.float32)
    lon = ds[lon_name].values
    lat = ds[lat_name].values
    ds.close()

    ocean = age[age > 0]
    print(f"  Age grid: shape={age.shape}, ocean mean={ocean.mean():.1f} Ma, "
          f"max={ocean.max():.1f} Ma")
    return {"age": age, "lon": lon, "lat": lat}


def load_globsed(filepath=None, region=None):
    """加载 GlobSed v3 全球海洋沉积厚度格网 (Straume et al. 2019).

    分辨率: 5 弧分 (~9 km at equator)
    值范围: 0-18000 m

    Returns dict with thickness, lon, lat.
    """
    if filepath is None:
        candidates = list(DATA_DIR.glob("**/GlobSed*/*.nc")) + \
                      list(DATA_DIR.glob("**/GlobSed*.nc"))
        if not candidates:
            raise FileNotFoundError(
                "GlobSed 文件未找到，请从 https://www.ngdc.noaa.gov/mgg/sedthick/ 下载")
        filepath = candidates[0]

    print(f"Loading GlobSed from: {filepath}")
    ds = xr.open_dataset(str(filepath), engine="h5netcdf")

    lon_name = "lon" if "lon" in ds.dims else "longitude"
    lat_name = "lat" if "lat" in ds.dims else "latitude"
    data_name = [v for v in ds.data_vars if len(ds[v].dims) >= 2][0]

    if region is not None:
        lon_min, lon_max, lat_min, lat_max = region
        ds_lon = ds[lon_name].values
        if lon_min < 0 and ds_lon.min() >= 0:
            lon_min += 360
            lon_max += 360
        ds = ds.sel(**{lon_name: slice(lon_min, lon_max),
                       lat_name: slice(lat_min, lat_max)})

    thickness = ds[data_name].values.astype(np.float32)
    lon = ds[lon_name].values
    lat = ds[lat_name].values
    ds.close()

    ocean = thickness[thickness > 0]
    print(f"  GlobSed: shape={thickness.shape}, "
          f"mean={ocean.mean():.0f}m, max={ocean.max():.0f}m")
    return {"thickness": thickness, "lon": lon, "lat": lat}


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
    """解析 MGD77 格式重力数据，返回经度、纬度、重力异常。

    MGD77 是固定列宽 FORTRAN 格式, 先用 read_fwf 按规范列宽解析,
    失败则回退到空白分割作为近似。
    """
    import pandas as pd
    try:
        df = pd.read_fwf(filepath, comment="#")
        return {"lon": df.iloc[:, 0].values, "lat": df.iloc[:, 1].values, "gravity": df.iloc[:, 2].values}
    except Exception:
        try:
            df = pd.read_csv(filepath, sep=r"\s+", comment="#")
            return {"lon": df.iloc[:, 0].values, "lat": df.iloc[:, 1].values, "gravity": df.iloc[:, 2].values}
        except Exception as e:
            print(f"  Warning: MGD77 gravity parse failed for {filepath}: {e}")
            return None


def _parse_mgd77_bathymetry(filepath):
    """解析 MGD77 格式水深数据，返回经度、纬度、水深值。"""
    import pandas as pd
    try:
        df = pd.read_fwf(filepath, comment="#")
        return {"lon": df.iloc[:, 0].values, "lat": df.iloc[:, 1].values, "bathymetry": df.iloc[:, 2].values}
    except Exception:
        try:
            df = pd.read_csv(filepath, sep=r"\s+", comment="#")
            return {"lon": df.iloc[:, 0].values, "lat": df.iloc[:, 1].values, "bathymetry": df.iloc[:, 2].values}
        except Exception as e:
            print(f"  Warning: MGD77 bathymetry parse failed for {filepath}: {e}")
            return None


if __name__ == "__main__":
    print("真实数据加载模块")
    print("=" * 50)
    print("使用前请确保数据已下载并放入 data/ 目录。")
    print("运行管线: python src/main.py")
