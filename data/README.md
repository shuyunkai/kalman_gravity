# 数据下载说明

数据文件因体积过大未上传至 GitHub，请从以下来源下载并放入对应目录。

## 目录结构与下载

### ssv32/ — SSV33.1 卫星重力场

| 文件 | 大小 | 下载地址 |
|------|------|----------|
| `grav_33.1.nc` | 599 MB | https://topex.ucsd.edu/pub/global_grav_1min/grav_33.1.nc |
| `curv_33.1.nc` | 640 MB | https://topex.ucsd.edu/pub/global_grav_1min/curv_33.1.nc |

### gebco/ — GEBCO 全球水深

| 文件 | 大小 | 下载地址 |
|------|------|----------|
| `GEBCO_2026_sub_ice/GEBCO_2026_sub_ice.nc` | 7.1 GB | https://download.gebco.net （选 GEBCO_2026 Grid, Bathymetry sub-ice, NetCDF, 全球范围） |

### shipborne/ — 船载重力与水深

| 来源 | 下载地址 |
|------|----------|
| MGDS (MGL1305, MGL1309 等) | https://www.marine-geo.org |
| NOAA NCEI | https://www.ncei.noaa.gov |

搜索航次名下载 Gravity Grid (.grd 或 .nc 格式)，放入对应航次子目录。

## 数据就绪检查

下载完成后运行 `python src/test_core.py`，若 5/5 通过则数据完整可用。
