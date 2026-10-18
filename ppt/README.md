# 汇报 PPT 工程

本项目汇报用的 PowerPoint 及其生成脚本。全部图件由 matplotlib 生成，可重跑复现。

## 目录

```
build_jq_ppt_v2.py          14 页汇报稿的生成脚本（python-pptx）
make_jq_ppt_figs.py         卡尔曼循环图、方法流程图、1′/15″ 对比、两种口径
make_jq_ppt_figs_v2.py      重力异常来源、测高反演流程、四个卡点、要素映射、本项目流程
make_jq_ppt_figs_extra.py   结果对比图、下一步路线图、封面研究区图
make_jq_ppt_figs_val.py     验证设计图、改善幅度图
make_jq_ppt_figs_compare.py 六类做法能力矩阵
figs/                       当前 14 页稿引用的 13 张图
重力场重建_杰青风格汇报.pptx  成稿
```

## 汇报结构（9 个章节，14 页）

| 页码 | 章节 | 内容 |
|---|---|---|
| — | 封面 | 研究区图 + 三个关键数字 |
| 2 | 1.1 | 什么是地球重力场（理论值 vs 实测值 → 重力异常 Δg） |
| 3 | 2.1 | 主流方法（一）：1′ 卫星测高重力模型 + 代表工作 + 共同特征 |
| 4 | 2.2 | 主流方法（二）：15″ 高分辨率重建 + 三条路线各自的局限 |
| 5 | 3.1 | 现有方法存在的问题：四个卡点 |
| 6 | 4.1 | 卡尔曼滤波能解决什么：卡点 → 方程要素 → 解决方式 |
| 7 | 5.1 | 什么是卡尔曼滤波：预测—加权校正循环 |
| 8 | 6.1 | 本项目的思路：三类观测写进同一套方程 |
| 9 | 7.1 | 本项目的卡尔曼滤波流程：先验→预测→卫星/船测校正→RTS→输出 |
| 10 | 8.1 | 验证设计：三个互不重叠的验证区、整条航次留出 |
| 11 | 8.2 | 结果（一）：三区横向对比 |
| 12 | 8.3 | 结果（二）：同一结果两种口径 |
| 13 | 8.4 | 结果（三）：改善幅度按基线分解 |
| 14 | 9.1 | 总结 |

## 复现

```bash
# 1. 先生成重力异常缩略缓存（fig_gravity 需要）
python - <<'PY'
import numpy as np, xarray as xr
from scipy.ndimage import zoom
from pathlib import Path
ds = xr.open_dataset("data/ssv32/grav_33.1.nc", engine="h5netcdf")
sub = ds["z"].sel(lon=slice(124, 153), lat=slice(12, 43)).values.astype(float)
ds.close()
Path("ppt/figs").mkdir(parents=True, exist_ok=True)
np.save("ppt/figs/_grav_cache.npy", zoom(sub, 0.5, order=1))
PY

# 2. 生成图件
python ppt/make_jq_ppt_figs.py
python ppt/make_jq_ppt_figs_v2.py
python ppt/make_jq_ppt_figs_extra.py
python ppt/make_jq_ppt_figs_val.py

# 3. 生成 PPT
python ppt/build_jq_ppt_v2.py
```

依赖：`python-pptx`、`matplotlib`、`numpy`、`scipy`、`xarray`、`h5netcdf`、`Pillow`。

## 注意

- 脚本中的路径按本机布局写死（`D:\gravity\...`），迁移时需改 `ROOT` / `FIGS` / `OUT`。
- PPT 字体为**微软雅黑**，需在 Windows 或装有该字体的环境打开，否则会回退字体。
- 图中数值来自 `output_real/multi_region_validation.txt`（15″ 点口径，整条航次留出）。
