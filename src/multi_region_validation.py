"""多区域独立航次验证 — 同一套方法/参数在多个海域上的泛化检验.

设计要点（与旧脚本的区别）:
  1. **整条航次留出**: 验证航次整体不参与解算（旧脚本是留一法, 每次只排除 1 条,
     其余 MR 航次仍在训练集里）。本脚本按区域显式指定 val_cruises,
     并把它们从训练点中彻底剔除, 真正做到"没加进解算的验证数据"。
  2. **多区域**: 用同一套参数（alpha/sigma/R/Q/迭代次数全部冻结）跑多个构造背景不同的
     海域, 看方法是否泛化, 而不是只在日本海沟 20°x16° 上成立。
  3. **两档口径**: 15″ 点对原始 10 s 点（含代表性误差）, 以及 1′ 块均值
     （与国际模型可比的"论文标准"口径）。
  4. **零船载区**: E 区（马里亚纳南部）区域内没有任何训练航次, 因此 KF 与双线性的
     唯一差别就是地形先验 —— 这是判定"地形物理先验是否真的有用"的干净实验。
  5. 统计口径: 逐点合并计算, Max|e| 取极值, 并报出 Bias / RMSE_bc(=STD)。

用法:
    python src/multi_region_validation.py --regions A G
    python src/multi_region_validation.py --alpha 0.2        # 全部区域
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from run_pipeline import load_ssv_region          # noqa: E402
from utils import cell_centers                    # noqa: E402
from three_method_compare import (                # noqa: E402
    load_bathy_15s, build_three_methods, compute_metrics,
    METHOD_BILINEAR, METHOD_FREQFUSION, METHOD_KF)
from independent_validation_v2 import (            # noqa: E402
    JP_DATA_DIR, load_cruise_points, assign_points_to_grid,
    )                                              # noqa: E402

OUTPUT_DIR = Path(__file__).parent.parent / "output_real"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UPSCALE = 4

# ---------------------------------------------------------------- 区域配置
# val  : 整条留出、不参与解算的验证航次
# train: 训练航次; None = 区域内除 val 之外的全部航次; () = 不使用任何船载训练数据
REGIONS = {
    "A_JapanTrench": dict(
        label="日本海沟北段", lon=(139.0, 142.0), lat=(34.0, 40.0), val=None,
        note="海沟轴部, 地形起伏最剧烈"),
    "B_HonshuEast": dict(
        label="本州东岸", lon=(139.0, 140.5), lat=(34.0, 35.6), val=None,
        note="小区域高密度重复测线, 多历元数据集"),
    "D_NankaiRyukyu": dict(
        label="南海-琉球", lon=(124.0, 132.0), lat=(24.0, 31.0),
        val=("YK14-14_gravity_dmo",),
        note="大陆边缘/琉球弧, 与海沟构造背景不同"),
    "F_PhilSeaMid": dict(
        label="菲律宾海中部", lon=(139.0, 152.0), lat=(13.0, 26.0),
        val=("YK18-08_gravity_dmo",),
        note="大洋盆地/岛弧过渡"),
    "E_MarianaSouth": dict(
        label="马里亚纳南部(零船载)", lon=(155.0, 167.0), lat=(-16.0, 8.0),
        val=("KR16-04_gravity_dmo",), train=(),
        note="区域内无训练航次 -> 隔离检验地形先验的贡献"),
    "G_CurrentBox": dict(
        label="当前验证区(参考)", lon=(132.0, 152.0), lat=(26.0, 42.0),
        val=("MR22-04_gravity_dmo", "MR22-02_gravity_dmo",
             "MR23-02_gravity_dmo", "MR21-02_gravity_dmo"),
        note="4 条 MR 航次同时留出 = 论文声称的设计"),
}

DEFAULT_VAL = ("MR22-04_gravity_dmo", "MR21-02_gravity_dmo",
               "MR23-02_gravity_dmo", "MR22-02_gravity_dmo")


def load_all_cruise_points(lon_range, lat_range):
    """读入 jp_data 下所有有数据的航次, 返回 {航次名: (N,3) 数组}."""
    out = {}
    for d in sorted(JP_DATA_DIR.iterdir()):
        if not d.is_dir():
            continue
        pts = load_cruise_points(d, lon_range, lat_range)
        if pts is not None and len(pts) >= 200:
            out[d.name] = pts
    return out


def build_region_data(cfg):
    """为一个区域准备好 build_three_methods 需要的 data 字典."""
    lon_range = tuple(cfg["lon"])
    lat_range = tuple(cfg["lat"])
    val_names = tuple(cfg.get("val") or DEFAULT_VAL)
    train_names_cfg = cfg.get("train", None)

    print(f"\n{'='*78}\n  [{cfg['label']}] lon={lon_range} lat={lat_range}\n"
          f"  {cfg['note']}\n{'='*78}")

    gravity_lr, _, lon_lr, lat_lr = load_ssv_region(lon_range, lat_range, load_vgg=False)
    ny_lr, nx_lr = gravity_lr.shape
    ny_hr, nx_hr = ny_lr * UPSCALE, nx_lr * UPSCALE
    lon_hr = cell_centers(lon_range[0], lon_range[1], nx_hr)
    lat_hr = cell_centers(lat_range[0], lat_range[1], ny_hr)
    print(f"  SSV 1': {gravity_lr.shape} -> 15'': ({ny_hr}, {nx_hr}) "
          f"= {ny_hr*nx_hr/1e6:.2f}M 像素")

    cruises = load_all_cruise_points(lon_range, lat_range)
    if not cruises:
        raise RuntimeError("区域内没有任何航次数据")
    val_names = tuple(n for n in val_names if n in cruises)
    if not val_names:
        raise RuntimeError(f"区域内没有可用的验证航次 (候选: {list(cruises)})")

    if train_names_cfg is None:
        train_names = tuple(n for n in cruises if n not in val_names)
    else:
        train_names = tuple(n for n in train_names_cfg if n in cruises)

    val_points = np.vstack([cruises[n] for n in val_names])
    print(f"  验证航次 ({len(val_names)}): "
          f"{', '.join(f'{n.split(chr(95))[0]}({len(cruises[n]):,})' for n in val_names)}"
          f"  -> {len(val_points):,} 点")
    if train_names:
        train_points = np.vstack([cruises[n] for n in train_names])
        print(f"  训练航次 ({len(train_names)}): {len(train_points):,} 点")
    else:
        train_points = np.zeros((0, 3))
        print("  训练航次: 无（零船载对照）")

    # --- 训练点 -> 15'' 网格 ---
    ship_grav = (assign_points_to_grid(train_points, lon_hr, lat_hr)
                 if len(train_points) else np.full((ny_hr, nx_hr), np.nan))
    n_ship = int(np.sum(~np.isnan(ship_grav)))
    print(f"  训练点落到 15'' 像素: {n_ship:,} ({n_ship/ship_grav.size*100:.2f}% 覆盖)")

    # --- 基准诊断（有意**不**修改船载数据）---
    # 原 three_method_compare.load_region_data 的规则是: 若 |mean(船载-SSV)|>=5 或
    # std>=15 mGal, 就给船载点加上 SSV 的长波趋势。该规则只对"残余重力"成立;
    # 在小框内, mean(船载-SSV) 会因为 SSV 不分辨短波(实测 std 常达 10 mGal 量级)
    # 而轻易超过 5 mGal, 此时再加一条 SSV 长波趋势会把数据彻底推歪(偏差反而放大几十 mGal)。
    # 本脚本因此只做诊断, 不改数据; 基准差异通过 Bias 列与 RMSE_bc(=STD) 两个口径如实呈现。
    ssv_hr = np.repeat(np.repeat(gravity_lr, UPSCALE, 0), UPSCALE, 1)
    train_mask = ~np.isnan(ship_grav)
    if n_ship:
        d0 = ship_grav[train_mask] - ssv_hr[train_mask]
        print(f"  训练点 - SSV: Bias={np.nanmean(d0):+.2f} "
              f"Median={np.nanmedian(d0):+.2f} STD={np.nanstd(d0):.2f} mGal"
              f"  (差异含基准项 + SSV 不分辨的短波项, 故只诊断不改正)")

    print("  加载 GEBCO 15'' 水深...")
    bathy_15s = load_bathy_15s(lon_range, lat_range, lon_hr, lat_hr)
    print(f"  GEBCO: [{np.nanmin(bathy_15s):.0f}, {np.nanmax(bathy_15s):.0f}] m")

    return dict(
        label=cfg["label"], note=cfg["note"], val_names=val_names,
        train_names=train_names, val_points=val_points,
        gravity_lr=gravity_lr, lon_lr=lon_lr, lat_lr=lat_lr,
        lon_hr=lon_hr, lat_hr=lat_hr,
        ny_lr=ny_lr, nx_lr=nx_lr, ny_hr=ny_hr, nx_hr=nx_hr,
        ship_grav=ship_grav, train_mask=train_mask,
        bathy_15s=bathy_15s, ssv_hr=ssv_hr,
        lon_range=lon_range, lat_range=lat_range)


def metrics_15s(model, val_points, lon_hr, lat_hr):
    """15″ 口径: 模型在验证点上的双线性取值 vs 船载原始 10 s 点。

    统一用 three_method_compare.compute_metrics 的键名 (RMSE/MAE/Max|e|/STD/Corr/
    Bias/n), 避免与 utils.compute_metrics 的键名 (Correlation/N, 且没有 Max|e|) 混用。
    """
    from scipy.interpolate import RegularGridInterpolator
    itp = RegularGridInterpolator((lat_hr, lon_hr), model,
                                  bounds_error=False, fill_value=np.nan)
    pred = itp(np.column_stack([val_points[:, 1], val_points[:, 0]]))
    truth = val_points[:, 2]
    m = compute_metrics(pred, truth)
    m["RMSE_bc"] = m["STD"]          # 去偏差 RMSE 恒等于 STD
    return m


def metrics_1min(model, val_points, lon_range, lat_range, upscale=UPSCALE):
    """1′ 块均值口径（与国际 1′ 模型可比的"论文标准"）。"""
    ny_hr, nx_hr = model.shape
    ny_lr, nx_lr = ny_hr // upscale, nx_hr // upscale
    model_1m = model[:ny_lr * upscale, :nx_lr * upscale].reshape(
        ny_lr, upscale, nx_lr, upscale).mean(axis=(1, 3))
    lon_lr = cell_centers(lon_range[0], lon_range[1], nx_lr)
    lat_lr = cell_centers(lat_range[0], lat_range[1], ny_lr)
    j = np.round((val_points[:, 0] - lon_lr[0]) / (lon_lr[1] - lon_lr[0])).astype(int)
    i = np.round((val_points[:, 1] - lat_lr[0]) / (lat_lr[1] - lat_lr[0])).astype(int)
    ok = (j >= 0) & (j < nx_lr) & (i >= 0) & (i < ny_lr)
    s = np.zeros((ny_lr, nx_lr)); c = np.zeros((ny_lr, nx_lr), dtype=int)
    np.add.at(s, (i[ok], j[ok]), val_points[ok, 2])
    np.add.at(c, (i[ok], j[ok]), 1)
    ship_1m = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    mask = np.isfinite(model_1m) & np.isfinite(ship_1m)
    m = compute_metrics(model_1m[mask], ship_1m[mask])
    m["RMSE_bc"] = m["STD"]
    return m, int(mask.sum())


def run_region(cfg, alpha=0.2, n_iter=6, smooth_sigma=15.0):
    t0 = time.time()
    data = build_region_data(cfg)
    print("  构建 3 方法 (双线性 / 频域融合 / KF)...")
    models, times = build_three_methods(data, alpha=alpha, n_iter=n_iter,
                                        smooth_sigma=smooth_sigma)
    val_points = data["val_points"]
    lon_hr, lat_hr = data["lon_hr"], data["lat_hr"]

    rows = []
    for name in (METHOD_BILINEAR, METHOD_FREQFUSION, METHOD_KF):
        grav = models[name]
        m15 = metrics_15s(grav, val_points, lon_hr, lat_hr)
        m1, n1 = metrics_1min(grav, val_points,
                              data["lon_range"], data["lat_range"], UPSCALE)
        rows.append(dict(method=name, m15=m15, m1=m1, n1=int(n1),
                         time=times[name]))
        print(f"    {name:<8} 15'': RMSE={m15['RMSE']:.3f} MAE={m15['MAE']:.3f} "
              f"STD={m15['STD']:.3f} Max|e|={m15['Max|e|']:.2f} "
              f"Bias={m15['Bias']:+.3f} | 1': RMSE={m1['RMSE']:.3f} (n={n1}) "
              f"| {times[name]:.0f}s")

    kf = next(r for r in rows if r["method"] == METHOD_KF)
    bi = next(r for r in rows if r["method"] == METHOD_BILINEAR)
    ff = next(r for r in rows if r["method"] == METHOD_FREQFUSION)
    print(f"    -> KF vs 双线性: 15'' {100*(1-kf['m15']['RMSE']/bi['m15']['RMSE']):+.1f}%  "
          f"1' {100*(1-kf['m1']['RMSE']/bi['m1']['RMSE']):+.1f}%   |   "
          f"KF vs 频域融合: 15'' {100*(1-kf['m15']['RMSE']/ff['m15']['RMSE']):+.1f}%")
    print(f"  用时 {time.time()-t0:.0f}s")

    return dict(label=data["label"], note=data["note"], rows=rows,
                n_val=len(val_points),
                n_train_px=int(data["train_mask"].sum()),
                n_train_cruises=len(data["train_names"]),
                val_names=data["val_names"], elapsed=time.time() - t0)


def write_summary(results, alpha, path):
    lines = []
    lines.append("=" * 108)
    lines.append("  多区域独立航次验证（整条航次留出，未参与解算）")
    lines.append(f"  KF 参数: alpha={alpha}, n_iter=6, smooth_sigma=15 px, "
                 f"Q=4, R_sat=9, R_ship=1 mGal^2（全部区域冻结同一套参数）")
    lines.append("  口径: 15'' = 模型点对船载原始 10 s 点；1' = 模型块均值对船载块均值")
    lines.append("=" * 108)
    for r in results:
        lines.append("")
        lines.append(f"[{r['label']}]  {r['note']}")
        lines.append(f"  验证航次({len(r['val_names'])}): {', '.join(r['val_names'])}"
                     f"   验证点 n={r['n_val']:,}")
        lines.append(f"  训练: {r['n_train_cruises']} 条航次 / {r['n_train_px']:,} 个 15'' 像素")
        lines.append(f"  {'方法':<10}{'RMSE15':>9}{'MAE15':>9}{'STD15':>9}"
                     f"{'Max|e|15':>10}{'Bias15':>9}{'Corr15':>9}"
                     f"{'RMSE1m':>9}{'MAE1m':>9}{'n1m':>8}")
        lines.append("  " + "-" * 100)
        for row in r["rows"]:
            a, b = row["m15"], row["m1"]
            lines.append(f"  {row['method']:<10}{a['RMSE']:>9.3f}{a['MAE']:>9.3f}"
                         f"{a['STD']:>9.3f}{a.get('Max|e|', float('nan')):>10.2f}"
                         f"{a['Bias']:>+9.3f}{a['Corr']:>9.4f}"
                         f"{b['RMSE']:>9.3f}{b['MAE']:>9.3f}{row['n1']:>8,}")
        kf = next(x for x in r["rows"] if x["method"] == METHOD_KF)
        bi = next(x for x in r["rows"] if x["method"] == METHOD_BILINEAR)
        ff = next(x for x in r["rows"] if x["method"] == METHOD_FREQFUSION)
        lines.append(f"  KF 改善: vs 双线性 15'' {100*(1-kf['m15']['RMSE']/bi['m15']['RMSE']):+.1f}%"
                     f" / 1' {100*(1-kf['m1']['RMSE']/bi['m1']['RMSE']):+.1f}%"
                     f"  |  vs 频域融合 15'' {100*(1-kf['m15']['RMSE']/ff['m15']['RMSE']):+.1f}%"
                     f" / 1' {100*(1-kf['m1']['RMSE']/ff['m1']['RMSE']):+.1f}%")
    lines.append("")
    lines.append("=" * 108)
    lines.append("  跨区域汇总（各区域 KF 的 15'' RMSE）")
    lines.append("=" * 108)
    lines.append(f"  {'区域':<24}{'KF RMSE':>10}{'双线性':>10}{'频域融合':>10}"
                 f"{'KF 改善':>10}{'验证点':>10}")
    lines.append("  " + "-" * 76)
    for r in results:
        kf = next(x for x in r["rows"] if x["method"] == METHOD_KF)
        bi = next(x for x in r["rows"] if x["method"] == METHOD_BILINEAR)
        ff = next(x for x in r["rows"] if x["method"] == METHOD_FREQFUSION)
        lines.append(f"  {r['label']:<24}{kf['m15']['RMSE']:>10.3f}{bi['m15']['RMSE']:>10.3f}"
                     f"{ff['m15']['RMSE']:>10.3f}"
                     f"{100*(1-kf['m15']['RMSE']/bi['m15']['RMSE']):>9.1f}%{r['n_val']:>10,}")
    txt = "\n".join(lines)
    path.write_text(txt, encoding="utf-8")
    print("\n" + txt)
    return txt


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--regions", nargs="*", default=None,
                   help="要跑的区域键名, 默认全部; 例: --regions A_JapanTrench E_MarianaSouth")
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--out", default="multi_region_validation.txt")
    a = p.parse_args()

    names = a.regions if a.regions else list(REGIONS)
    results = []
    for k in names:
        if k not in REGIONS:
            print(f"  跳过未知区域 {k}; 可选: {list(REGIONS)}")
            continue
        try:
            results.append(run_region(REGIONS[k], alpha=a.alpha))
        except Exception as e:
            import traceback
            print(f"  !! 区域 {k} 失败: {type(e).__name__}: {e}")
            traceback.print_exc()

    if results:
        write_summary(results, a.alpha, OUTPUT_DIR / a.out)
        print(f"\n  汇总写入: {OUTPUT_DIR / a.out}")


if __name__ == "__main__":
    main()
