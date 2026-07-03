# M2 地形先验 KF — 15'' 重力场重建解算流程

## 整体流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        输入数据 (3 个独立源)                          │
├─────────────────────────────────────────────────────────────────────┤
│  ① SSV33.1 卫星测高重力 1'   ② GEBCO 2026 海底地形 15''            │
│     (data/ssv32/)              (data/gebco/)                        │
│                                                                       │
│  ③ 日本船载重力 30+ 航次 10s 原始点 (jp_data/*/)                     │
│     切分: 4 条 MR 系列作独立验证, 其余 26 条作训练                   │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          数据预处理                                  │
├─────────────────────────────────────────────────────────────────────┤
│  1. SSV 1' 加载 + 经纬度切片 (load_ssv_region)                       │
│  2. GEBCO 15'' 加载 + 0.5° padding (load_gebco_region)               │
│  3. 船载 .grv 解析 → (lon, lat, faa) 数组 (load_all_cruises_split)   │
│  4. 船载点精确像素分配到 15'' 网格 (assign_points_to_grid)           │
│  5. 趋势恢复: 船载与 SSV 系统偏差 > 5 mGal 时用 SSV 平滑趋势补偿    │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    M2 物理先验构建 (核心)                            │
├─────────────────────────────────────────────────────────────────────┤
│  步骤 A: SSV 1' 双线性上采样到 15'' (作低频底色 ssv_hr)              │
│  步骤 B: GEBCO 水深 → 布格重力异常 (一阶布格近似)                    │
│            g_bouguer = 2π·G·ρ·depth × 1e5 (mGal)                    │
│            G=6.6743e-11, ρ=2670 kg/m³, depth=bathy (m)              │
│  步骤 C: 布格重力高斯高通滤波 (cutoff=0.05° ≈ 5.5km)                │
│            bathy_hp = bathy_bouguer - gaussian_filter(bathy_bouguer)│
│  步骤 D: 物理先验组合                                                │
│            prior = ssv_hr + α × bathy_hp   (α=0.2, 大区域最优)       │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│              卡尔曼滤波后验修正 (KalmanGravityReconstructor)         │
├─────────────────────────────────────────────────────────────────────┤
│  状态变量 x: 15'' 高分辨率格网 (ny_hr × nx_hr, 18.4M 像素)           │
│  初始状态: x_0 = prior (物理先验, 不用 SSV 双线性直接初始化)         │
│                                                                       │
│  两个观测方程:                                                        │
│    ① 卫星块均值约束: z_sat = H_sat · x, R_sat=9 (每 4×4 块均值=SSV)  │
│    ② 船载点观测:    z_ship = H_ship · x, R_ship=1 (训练点位置)       │
│                                                                       │
│  6 次 KF 迭代 (predict-update-smooth):                              │
│    for iter in range(6):                                             │
│      1. predict:    x_pred = F·x   (F = 高斯平滑, σ=15 像素)         │
│      2. update_sat: 在卫星 1' 块上做贝叶斯更新                       │
│      3. update_ship:在船载点位置做贝叶斯更新                         │
│    末尾 RTS 后向平滑 (smooth_backward)                               │
│                                                                       │
│  输出: 15'' 重力场 gravity_15s (ny_hr × nx_hr)                       │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       独立验证 (MR22-04)                             │
├─────────────────────────────────────────────────────────────────────┤
│  1. 模型场在 MR22-04 航迹点上双线性插值取值                          │
│  2. 与船载真值对比, 计算 RMSE/MAE/STD/Corr/Bias/Max|e|              │
│  3. 多航次重复 (MR22-04/02, MR23-02, MR21-02)                       │
│  4. 多尺度验证 (20°×16° / 30°×24°)                                  │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
                  15'' 重力场模型 + 精度报告
```

## 关键数学公式

### 1. 布格重力近似 (一阶 Parker)

从海底地形 h(x,y) 计算海面观测的布格重力异常:

```
g_bouguer = 2π·G·ρ·h    [m/s²]
         = 2π·G·ρ·h × 10^5    [mGal]
```

其中:
- G = 6.6743×10⁻¹¹ m³ kg⁻¹ s⁻² (万有引力常数)
- ρ = 2670 kg/m³ (地壳平均密度)
- h = -bathy (海底深度, m, 正向下)
- 系数 2π·G·ρ·10⁵ ≈ 0.1118 mGal/m

**注**: 完整 Parker 频域正演含 exp(-|k|·d) 向下延拓衰减, 一阶近似 (d=0) 等价于布格平板公式. M2 用一阶近似已足够, 因 KF 会用船载观测修正短波长偏差.

### 2. M2 物理先验

```
prior(x,y) = ssv_hr(x,y) + α × HP[bathy_bouguer](x,y)
```

- ssv_hr: SSV 1' 双线性上采样到 15'' (低频底色, ≤30 km 波长)
- HP[·]: 高斯高通滤波, cutoff=0.05° (≈5.5 km), 保留中频地形信号
- α=0.2: 大区域最优权重 (10°×10° 时 α=0.5, 大区域需减小避免偏差累积)

### 3. 卡尔曼滤波状态空间

**状态方程**:
```
x_{k+1} = F · x_k + w_k,    w_k ~ N(0, Q)
```
F = 高斯空间平滑算子 (σ=15 像素 ≈ 7 km), Q = 4 mGal² (过程噪声)

**观测方程 1 — 卫星块均值**:
```
z_sat = H_sat · x + v_sat,    v_sat ~ N(0, R_sat)
```
H_sat: 4×4 块均值算子 (15''→1' 降采样), R_sat = 9 mGal²

**观测方程 2 — 船载点**:
```
z_ship = H_ship · x + v_ship,    v_ship ~ N(0, R_ship)
```
H_ship: 点采样算子 (训练点位置), R_ship = 1 mGal²

**KF 迭代**:
```
predict:  x_pred = F · x
          P_pred = F · P · F^T + Q

update:   K = P_pred · H^T · (H · P_pred · H^T + R)^{-1}
          x = x_pred + K · (z - H · x_pred)
          P = (I - K · H) · P_pred
```

每轮迭代包含 1 次 predict + 1 次 update_sat + 1 次 update_ship, 共 6 轮. 末尾 RTS 后向平滑得到最终状态估计.

## 文件结构

```
src/
├── kalman_gravity.py          # KalmanGravityReconstructor (核心 KF 类)
├── kf_novel.py                # BathyPriorKF (M2 主方法)
├── freq_fusion_baseline.py    # 频域融合基线 + butterworth_2d_filter
├── independent_validation_v2.py  # 多航次数据加载 + 验证
├── run_pipeline.py            # SSV/GEBCO 数据加载器
├── utils.py                   # 通用工具 (metrics, radial_power_spectrum 等)
├── val_visualization.py       # 验证可视化 (散点/直方/QQ/剖面等)
├── compare_m2_traditional.py  # 3 方法对比主脚本
├── tune_m2_alpha.py           # α 参数扫描
├── m2_multi_cruise_val.py     # 多航次验证
├── m2_distinction_viz.py      # 本质区别可视化 (差异场/功率谱)
├── m2_large_region_val.py     # 30°×24° 大海域验证
└── m2_accuracy_comprehensive.py  # 综合精度对比 (6 张图)
```

## 复现命令

```bash
cd src

# 1. α 扫描 (10 分钟) — 找最优 α
python tune_m2_alpha.py

# 2. 多航次验证 (5 分钟) — 4 条 MR 系列航次
python m2_multi_cruise_val.py --alpha 0.2

# 3. 本质区别可视化 (3 分钟) — 4 张图
python m2_distinction_viz.py --alpha 0.2

# 4. 30°×24° 大海域 (5 分钟)
python m2_large_region_val.py --alpha 0.2

# 5. 综合精度对比 (3 分钟) — 6 张精度指标图
python m2_accuracy_comprehensive.py --alpha 0.2

# 6. 单次 3 方法对比 (2 分钟)
python compare_m2_traditional.py
```

## 关键参数

| 参数 | 值 | 含义 |
|------|-----|------|
| α (alpha) | 0.2 | 地形先验权重 (20°×16° 最优) |
| bathy_cutoff_deg | 0.05° | 布格高通 cutoff ≈ 5.5 km |
| n_iter | 6 | KF 迭代次数 |
| smooth_sigma | 15 像素 | KF 状态空间平滑 σ ≈ 7 km |
| Q (过程噪声) | 4 mGal² | KF 过程噪声方差 |
| R_sat | 9 mGal² | 卫星观测噪声方差 |
| R_ship | 1 mGal² | 船载观测噪声方差 |
| upscale | 4 | 1' → 15'' 上采样倍数 |
