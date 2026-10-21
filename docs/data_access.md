# 数据获取操作手册

需要账号/申请的数据，按下面步骤操作；不需要账号的（GEM、GNS、CGS、GMRT）我已经下好。

## 1. SWOT L2 SSH（补 2–5 km 重力）

1. 注册 Earthdata：<https://urs.earthdata.nasa.gov/users/new>
2. 登录后到 “Applications” 授权 **PO.DAAC**；
3. 安装：`pip install earthaccess`
4. 运行：
   ```bash
   python src/download_swot.py --bbox 141 32 149 40 --start 2023-08-01 --end 2024-06-30
   ```
5. 首次会弹浏览器登录；之后自动下载到 `data/swot`。

产品：`SWOT_L2_LR_SSH_EXPERT`（沿轨约 2 km）。若搜不到，改 `--short-name SWOT_L2_LR_SSH_BASIC`。

## 2. 船测重力

### NCEI GEODAS
- <https://www.ncei.noaa.gov/maps/geophysics/>
- 选 “Marine Trackline Geophysical Data” → 勾 Gravity → 框选 141–149°E、32–40°N → 下载 MGD77。

### JODC（日本周边更全，需申请）
- <https://www.jodc.go.jp/>
- 注册后选“重力データ”，框选区域，提交申请；一般邮件回复下载链接。

### MGDS（补充）
- <https://www.marine-geo.org/> → 搜 Gravity → 框选区域 → 下载。

## 3. 海域断层

### BOEM（墨西哥湾）
- <https://www.data.boem.gov/> → GIS Data → Faults/Geology shapefile。

### GNS 新西兰（已可下载）
- <https://gis.gns.cri.nz/server/rest/services/af250/MapServer>
- 需要扩窗时告诉我经纬度，我直接取 GeoJSON。

### 日本 AIST/NIED（需申请）
- <https://gbank.gsj.jp/activefault/> → 在线检索 → “お問い合わせ”申请矢量数据。

### 意大利 ITHACA（需申请）
- <https://hub.geosciences-ir.it/catalogo_dati/dataset/ithaca-catalogo-delle-faglie-capaci-geosciences-ir>

## 4. 下载后放置

```
D:\gravity\kalman_gravity\data\
├── swot\
├── ship_gravity\
└── faults\
```

放好后告诉我，我接入 KF 观测项并重跑 2–5 km 断层验证。
