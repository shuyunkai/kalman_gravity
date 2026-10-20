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
| 线性体提取 | THG 84% 阈值（仅海洋）→ Zhang–Suen 细化 → 去小块/闭环（细长比>2.2） |

**结果**：

- `fig_scale_tilt.png`：三个尺度带的场与 tilt 属性
- `fig_fault_candidates.png`：三层线性体叠加
- `fig_fault_rose.png`：走向玫瑰图
- `fig_fault_validation.png`：线性体 + USGS M≥4 地震 + 距离分布

| 尺度带 | 线性体像元 | 地震中位距 | 20 km 内占比 | 随机点基线 |
|---|---|---|---|---|
| 10–50 km | 4,159（0.11%） | 39.1 km | **24.2%** | 6.1% |
| 50–100 km | 5,982（0.16%） | 123.9 km | 9.8% | 11.3% |
| 100–200 km | 2,180（0.06%） | 356.0 km | 0% | 3.2% |

海沟轴到最近线性体：中位 91.4 km，P90 151.7 km。

**解读**：只有 10–50 km 这一层相对随机点有约 **4 倍富集**，说明浅部线性体与地震活动有一定空间关联；
50–100 km 与随机持平，100–200 km 因线性体太少而失效。也就是说，**尺度分解能提供候选，但目前还不足以直接当断层图用**。

## 三、Parker–Oldenburg 反演海底地形（新增）

**正演**（Parker, 1973）：

Δg(k) = 2πGρ·e^(−k z₀)·Σ_{n=1}^{N} (−k)^(n−1)/n!·F[hⁿ]

**反演**（Oldenburg, 1974）：

h ← h + relax·LP[ F⁻¹( (F[Δg_obs]−F[Δg_pred])·e^(+k z₀) / (2πGρ) ) ]

参数：z₀=5062 m，ρ 扫描 1640/1800/2000/2200（选 2200），带通 20–200 km，LP 25 km，relax=0.6，8 轮。

**结果**（`fig_bathymetry_inversion.png`、`fig_inversion_eval.png`）：

| 指标 | 值 |
|---|---|
| 正演校验（GEBCO 带通地形 → Δg） | corr 0.759，振幅比 1.277 |
| 反演 vs GEBCO 带通地形 | RMSE 651 m，corr 0.610 |
| 振幅标定斜率（真值/反演） | 1.34（反演振幅偏小约 25%） |
| 残差收敛 | 24.85 → 0.88 mGal |

分频段：

| 频段 | RMSE | corr | 振幅比 |
|---|---|---|---|
| 20–50 km | 244 m | 0.44 | 0.60 |
| 50–100 km | 288 m | 0.54 | 0.49 |
| 100–200 km | 355 m | 0.75 | 0.40 |

剖面（143.5°E，39°→33°N）显示海沟、海山位置恢复良好，振幅整体偏小。

## 复现

```bash
python src/prepare_test_field.py     # 生成测试区 15″ 场（约 1 分钟）
python src/run_three_parts.py        # 尺度分解 + 断裂验证 + 反演（约 4 分钟）
```

输出：`output_real/three_parts/`（6 张图 + report.txt）。

## 参考

- Parker, R.L., 1973. The rapid calculation of potential anomalies. *Geophys. J. Int.* 31, 447–455.
- Oldenburg, D.W., 1974. The inversion and interpretation of gravity anomalies. *Geophysics* 39, 526–536.
- pyParkO: <https://github.com/Eurrreka/pyParkO>（Parker–Oldenburg 的 Python 实现，含变密度）
- Cordell, L., Grauch, V.J.S., 1985. Mapping basement magnetization zones from aeromagnetic data.
- Miller, H.G., Singh, V., 1994. Potential field tilt — a new concept for location of potential field sources.
- Wijns, C., et al., 2005. Theta map: edge detection in magnetic data.
- Blakely, R.J., 1995. *Potential Theory in Gravity and Magnetic Applications*.

## 已知局限（写论文时要交代）

1. 断裂候选是**边缘/线性体**，只做了地震点富集检验；要称“识别断裂”，还需与公开断裂图、地震剖面、船测地形逐条对比；
2. 线性体很稀疏（0.06–0.16%），阈值与细长比筛选对结果影响大，需要做敏感性分析；
3. 反演用常密度，振幅偏小约 25%，未考虑沉积层与变密度；下一步用 pyParkO 的 VDM。
4. 频域方法有边缘效应，正式结果应裁边或做 taper。
