# 卡尔曼滤波全球重力场建模与海底地形反演

基于卡尔曼滤波，融合卫星测高与船载重力数据，构建全球海洋重力场模型并反演海底地形。

## 环境要求

```bash
pip install numpy scipy matplotlib xarray netCDF4 h5netcdf requests cartopy pandas python-docx
```

## 代码目录

```
kalman_gravity/
├── README.md                  项目说明
│
├── src/                       源码 (9个文件)
│   ├── main.py                统一命令行入口
│   ├── kalman_gravity.py      核心算法 (KF重力重建 + EnKF + Parker水深反演)
│   ├── run_pipeline.py        区域真实数据完整流程
│   ├── run_global.py          全球模型构建 + 超分辨率
│   ├── make_figures.py        分析图表生成 (fig1~7)
│   ├── utils.py               工具函数 (评估指标/VGG/滤波/插值/绘图)
│   ├── real_data.py           真实数据加载接口 (SSV/GEBCO/船载)
│   ├── test_core.py           核心算法单元测试 (5项)
│   └── __init__.py
│
├── data/                      全部原始数据
│   ├── ssv32/                 SSV33.1 卫星重力 (grav + curv, 1.2 GB)
│   ├── gebco/                 GEBCO 全球水深 (sub_ice + surface + TID, 17.8 GB)
│   └── shipborne/             船载重力+水深 (6个航次, 18个俯冲带)
│
├── output_global/             全球模型输出
│   └── kalman_gravity_1min.nc 全球1'卡尔曼滤波重力场 (1.2 GB)
│
├── output_figures/            分析图表 (7张)
│   ├── fig1 全球重力场        SSV33.1 vs KF vs 差异
│   ├── fig2 船载数据分布      全球船载重力数据覆盖
│   ├── fig3 超分辨率对比      日本海沟六图对比
│   ├── fig4 验证分析          散点图+误差直方图+残差+提升量
│   ├── fig5 功率谱            SSV vs KF 径向功率谱
│   ├── fig6 水深反演          日本海沟 GEBCO vs KF vs Parker
│   └── fig7 汇总表            模型参数对比
│
└── output_real/               区域结果图
    ├── real_gravity.png       重力重建对比
    ├── real_bathymetry.png    水深反演对比
    └── real_residual.png      水深残差图
```

## 数据准备

运行前需将数据放入 `data/` 目录：

| 数据 | 子目录 | 必需文件 | 下载地址 |
|------|--------|----------|----------|
| SSV33.1 卫星重力 | `data/ssv32/` | `grav_33.1.nc`, `curv_33.1.nc` | https://topex.ucsd.edu/pub/global_grav_1min/ |
| GEBCO 全球水深 | `data/gebco/` | `GEBCO_2026_sub_ice.nc` | https://download.gebco.net |
| 船载重力 | `data/shipborne/` | *.grd.gz 格网文件 | https://www.marine-geo.org |

## 运行步骤

**第一步：单元测试（确认环境正常）**
```bash
python src/test_core.py
```
通过则输出 `5/5 tests passed`。

**第二步：构建全球重力场模型**
```bash
python src/main.py --global-model
```
输出 `output_global/kalman_gravity_1min.nc`（1.2 GB），全球 1 弧分卡尔曼滤波重力场，同时船载数据覆盖区域自动做超分辨率重建。

**第三步：区域管线（日本海沟示例）**
```bash
python src/main.py --upscale 4
```
输出 `output_real/` 下 3 张图，展示区域重力重建和水深反演结果。

**第四步：生成全部分析图表**
```bash
python src/main.py --figures
```
输出 `output_figures/` 下 7 张分析图。

## 结果说明

### 全球模型 (`output_global/`)

| 文件 | 内容 |
|------|------|
| `kalman_gravity_1min.nc` | 全球 1' 卡尔曼滤波重力场 NetCDF，含 `gravity_anomaly`(mGal) 和 `vgg`(Eötvös) 两个变量，格网 21600×9600 |

### 区域结果 (`output_real/`)

| 文件 | 内容 |
|------|------|
| `real_gravity.png` | 三图并排：左=卫星重力(1')、中=卡尔曼滤波(15'')、右=双线性插值(15'') |
| `real_bathymetry.png` | 三图并排：左=GEBCO 先验、中=卡尔曼滤波反演、右=GEBCO 真值 |
| `real_residual.png` | KF 反演水深与 GEBCO 的差值空间分布 |

### 分析图表 (`output_figures/`)

| 文件 | 内容 |
|------|------|
| `fig1_global_gravity.png` | 全球重力场：SSV33.1 原始 / KF 重建 / 差异图 |
| `fig2_shipborne_distribution.png` | 全球船载重力数据覆盖分布（含海岸线底图） |
| `fig3_japan_super_resolution.png` | 日本海沟 2×3 六图：卫星/双线性/KF/船载/残差对比 |
| `fig4_scatter_histogram.png` | 2×2 四图：散点对比、误差直方图、空间残差、KF 精度提升量 |
| `fig5_power_spectrum.png` | 径向功率谱：SSV33.1 原始 vs KF 重建 |
| `fig6_bathymetry_inversion.png` | 2×3 六图：GEBCO/KF/Parker 水深反演 + 残差 + 重力场 |
| `fig7_summary_table.png` | 模型参数与验证精度汇总表 |

## 数据来源

| 数据 | 来源 | 下载地址 |
|------|------|----------|
| SSV33.1 卫星重力 | SIO/UC San Diego | https://topex.ucsd.edu/pub/global_grav_1min/ |
| GEBCO 全球水深 | GEBCO Seabed 2030 | https://download.gebco.net |
| 船载重力 | NOAA NCEI / MGDS | https://www.marine-geo.org |
