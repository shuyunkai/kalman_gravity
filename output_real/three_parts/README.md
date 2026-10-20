# 三部分工作：重建 / 尺度分解与断裂 / 反演

对应开题调整后的三条研究内容。测试区：**141–149°E，32–40°N**（日本海沟—伊豆小笠原北部），
15″ 网格 1920×1920（dx≈0.38 km，dy≈0.46 km）。

## 一、15″ 重力场重建（已有）

`BathyPriorKF`：SSV 1′ 块填充 + α(0.5)×地形布格高频作先验，6 轮 KF（卫星块均值 + 船载点更新）+ RTS。
测试区场由 `src/prepare_test_field.py` 生成并缓存为 `output_real/test_field_15s.npz`。

## 二、多尺度分解与断裂识别（新增）

**方法**（频域经典算子）：

| 步骤 | 公式/实现 |
|---|---|
| 尺度分解 | 带通 10–50 / 50–100 / 100–200 km（Butterworth，4 阶） |
| 水平总梯度 THG | √((∂g/∂x)²+(∂g/∂y)²)，Cordell & Grauch (1985) |
| 垂向导数 | F⁻¹[|k|·F[g]] |
| tilt | atan2(∂g/∂z, THG)，Miller & Singh (1994) |
| theta map | acos(THG/√(THG²+gz²))，Wijns et al. (2005) |
| 线性体提取 | THG 86% 阈值 → Zhang–Suen 细化 → 去小块/闭环（细长比>2.2） |

**结果**（`output_real/three_parts/`）：

- `fig_scale_tilt.png`：三个尺度带的场与 tilt 属性
- `fig_fault_candidates.png`：三层线性体叠加（黑线）
- 线性体像元：10–50 km 10,130；50–100 km 5,359；100–200 km 2,727

**解读**：日本海沟轴部（142–144°E）在三个尺度上都呈连续线性梯度带；海山处多为闭合环，已被细长体筛选剔除；
左上角陆坡与图幅边缘仍有伪线，需在正式结果中裁边或加 taper。

## 三、Parker–Oldenburg 反演海底地形（新增）

**正演**（Parker, 1973）：

Δg(k) = 2πGρ·e^(−k z₀)·Σ_{n=1}^{N} (−k)^(n−1)/n!·F[hⁿ]

**反演**（Oldenburg, 1974）：

h ← h + relax·LP[ F⁻¹( (F[Δg_obs]−F[Δg_pred])·e^(+k z₀) / (2πGρ) ) ]

参数：z₀=5062 m（平均水深），ρ 扫描 1640/1800/2000/2200（选 2200），
带通 20–200 km，LP 截止 25 km，relax=0.6，8 次迭代，边缘余弦 taper。

**结果**（`fig_bathymetry_inversion.png`）：

| 指标 | 值 |
|---|---|
| 正演校验（GEBCO 带通地形 → Δg） | corr 0.759，振幅比 1.277 |
| 反演 vs GEBCO 带通地形 | RMSE 651 m，corr 0.610 |
| 残差收敛 | 24.85 → 0.88 mGal（8 轮） |

**解读**：海沟、海山链的位置与形态恢复较好；振幅整体偏小（回归斜率约 0.5），
主要因为常密度假设 + 低通阻尼，未考虑沉积层与地壳结构。下一步应换变密度（pyParkO 的 VDM）并做振幅标定。

## 复现

```bash
python src/prepare_test_field.py     # 生成测试区 15″ 场（约 1 分钟）
python src/run_three_parts.py        # 尺度分解 + 断裂候选 + 反演（约 3 分钟）
```

输出：`output_real/three_parts/`（3 张图 + report.txt）。

## 参考

- Parker, R.L., 1973. The rapid calculation of potential anomalies. *Geophys. J. Int.* 31, 447–455.
- Oldenburg, D.W., 1974. The inversion and interpretation of gravity anomalies. *Geophysics* 39, 526–536.
- pyParkO: <https://github.com/Eurrreka/pyParkO>（Parker–Oldenburg 的 Python 实现，含变密度）
- Cordell, L., Grauch, V.J.S., 1985. Mapping basement magnetization zones from aeromagnetic data.
- Miller, H.G., Singh, V., 1994. Potential field tilt — a new concept for location of potential field sources.
- Wijns, C., et al., 2005. Theta map: edge detection in magnetic data.
- Blakely, R.J., 1995. *Potential Theory in Gravity and Magnetic Applications*.

## 已知局限（写论文时要交代）

1. 断裂候选是**边缘/线性体**，不是地质断层；需与公开断裂图、地震剖面、船测地形对比后才能称“识别断裂”；
2. 反演用常密度 1640/2200，未考虑沉积层与变密度；
3. KF 场本身有 ~4 mGal 误差，会传入反演；
4. 频域方法有边缘效应，正式结果应裁边或做 taper。
