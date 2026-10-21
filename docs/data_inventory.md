# 数据清单与获取状态

## 已下载（无需账号）

| 数据 | 位置 | 说明 |
|---|---|---|
| SSV33.1 1′ 重力 | `data/ssv32/grav_33.1.nc` | 原有 |
| GEBCO 15″ 地形 | `data/gebco/` | 原有 |
| GMRT 区域地形 | `output_real/three_parts/gmrt_region.grd` | 141–149°E/32–40°N，低分辨率 |
| GMRT GM9 高分辨率地形 | `output_real/three_parts/gmrt_gm9_max.grd` | 约 40 m 网格 |
| GEM 全球活动断层 | `output_real/three_parts/gem_active_faults.geojson` | 日本 45 条、爱琴海 161 条 |
| GNS 新西兰断层 | `output_real/three_parts/nz_faults.json`、`nz_gm9_offshore.json` | af250、GM9 海域 |
| CGS 加州断层 | `output_real/three_parts/ca_faults.json` | 233 条 |
| PB2002 板块边界 | `output_real/three_parts/pb2002.json` | 3 条 |
| SWOT L2 LR SSH | `data/swot/*.nc` | 2024-01–03，100 个 granule（下载中） |

## 需要你注册/申请

| 数据 | 入口 | 需要什么 |
|---|---|---|
| SWOT（已完成） | https://urs.earthdata.nasa.gov/ | 免费账号 + token（已提供） |
| NCEI 船测重力 | https://www.ncei.noaa.gov/maps/geophysics/ | 一般无需登录，但本机网络不通；需你在浏览器下载 MGD77 |
| JODC 船测重力 | https://www.jodc.go.jp/ | 注册 + 数据申请（邮件回复） |
| AIST/NIED 活断层 | https://gbank.gsj.jp/activefault/ | 在线检索 + 矢量数据申请 |
| BOEM 海域断层 | https://www.data.boem.gov/ | 公开数据，本机网络不通；需浏览器下载 |
| ITHACA 意大利断层 | https://hub.geosciences-ir.it/ | 页面申请/联系 ISPRA |
| 航空重力 | 各国地调局 | 合作/申请，通常不公开 |

## 放置约定

```
D:\gravity\kalman_gravity\data\
├── swot\            # SWOT L2 SSH（已在此）
├── ship_gravity\    # NCEI/JODC 下载后放这里
└── faults\          # BOEM/NIED/ITHACA 下载后放这里
```

放好后告诉我，我接入 KF 观测项并重跑 2–5 km 验证。

## SWOT 下载结果（2024-01 至 02-20）

- 50 个 Basic granule，共 556.8 MB（含少量 Expert/WindWave）；
- 日本窗口有效像素 613,448 个，覆盖 141–149°E、32–40°N；
- 时间跨度 2024-01-01 至 2024-02-20，共 24 个观测日；
- SSH 12.9–38.4 m，SSHA ±2 m，测高不确定度中位 3.0 mm。

CMR 单页上限 200 条（=50 个 granule × 4 产品），3 月的数据还需翻页继续下。


## SWOT 下载完成（2024-01–06）

- 180 个 Basic granule，1.63 GB；
- 日本窗口有效像素 2,094,603；
- SSH 12.7–37.9 m，SSHA ±2.5 m，测高不确定度中位 2.8 mm；
- 缓存 data/swot/swot_japan_subset.npz。

