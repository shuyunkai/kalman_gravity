"""全球 15″ 重力场产品生成 v2（分级产品, 可复现）.

为什么要重做:
  旧 output_global/kalman_gravity_15s.nc 由已删除的脚本产生, 且实测存在三个问题:
    1) 坐标用 np.linspace(lat1[0], lat1[-1], 38400) 生成 —— 格点约定, 与 SSV/GEBCO 的
       像元中心约定不一致, 步长实测 0.00416634 度 而非 1/240=0.004166667 度;
    2) 船载覆盖区被贴入"残余重力"结果而没有做基准恢复: 与 SSV 相比, 日本海沟一带
       bias=-17.9 mGal、南海 bias=-17.6 mGal, 最差 4x4 度块 bias=-28 mGal;
       全网格 5.5% 的像元与 SSV 差 >2 mGal;
    3) 只有 gravity_anomaly 一个变量, 无数据来源标识, long_name 自称 "upsampled"。

v2 产品定义（分级 + 来源可追溯）:
  * 网格: 15″, 像元中心约定, lat -80..80 (38400), lon -180..180 (86400)
  * gravity_anomaly: 基底 = SSV33.1 1' 场的 4x 双线性上采样（像元中心对齐）;
                     在船载覆盖区用 BathyPriorKF（主方法, 已修全部已知 bug）的 15″ 解覆盖
  * source (int8): 0 = 上采样基底, 1 = KF 15″ 解算
  * 逐纬度带流式写出（峰值内存 <1 GB）

用法:
    python src/run_global_v2.py
    python src/run_global_v2.py --regions G_CurrentBox
    python src/run_global_v2.py --out test.nc --band 480
"""
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import zoom
import netCDF4 as nc

sys.path.insert(0, str(Path(__file__).parent))

from run_pipeline import DATA_DIR, NC_ENGINE          # noqa: E402
from utils import cell_centers                        # noqa: E402
from kf_novel import BathyPriorKF                     # noqa: E402
from three_method_compare import load_bathy_15s       # noqa: E402
from multi_region_validation import (                 # noqa: E402
    load_all_cruise_points, assign_points_to_grid, REGIONS)

OUT = Path(__file__).parent.parent / "output_global"
OUT.mkdir(parents=True, exist_ok=True)

LAT0, LAT1 = -80.0, 80.0
LON0, LON1 = -180.0, 180.0
NLAT, NLON = 38400, 86400            # 15 arc-sec
UPSCALE = 4
ALPHA = 0.5
N_ITER = 6
SMOOTH_SIGMA = 4.0

# 默认参与解算的区域（边界必须都是 15″ 的整数倍, 才能与全球网格整数对齐）
DEFAULT_REGIONS = ("G_CurrentBox", "A_JapanTrench", "B_HonshuEast",
                   "D_NankaiRyukyu", "F_PhilSeaMid", "E_MarianaSouth")


def solve_region(cfg, verbose=True):
    """在一个区域内用全部船载数据做 15″ KF 解算.

    返回 (gravity_15s, lon_hr, lat_hr, 诊断 dict)。
    """
    lon_range = tuple(cfg["lon"])
    lat_range = tuple(cfg["lat"])
    ds = xr.open_dataset(str(DATA_DIR / "ssv32" / "grav_33.1.nc"), engine=NC_ENGINE)
    glr = ds["z"].sel(lon=slice(lon_range[0], lon_range[1]),
                      lat=slice(lat_range[0], lat_range[1])).values.astype(np.float64)
    ds.close()
    ny_lr, nx_lr = glr.shape
    ny_hr, nx_hr = ny_lr * UPSCALE, nx_lr * UPSCALE
    lon_hr = cell_centers(lon_range[0], lon_range[1], nx_hr)
    lat_hr = cell_centers(lat_range[0], lat_range[1], ny_hr)

    cruises = load_all_cruise_points(lon_range, lat_range)
    if not cruises:
        raise RuntimeError("区域内没有船载数据")
    pts = np.vstack(list(cruises.values()))
    ship = assign_points_to_grid(pts, lon_hr, lat_hr)
    train_mask = ~np.isnan(ship)
    n_px = int(train_mask.sum())
    if verbose:
        print(f"    {cfg['label']}: {len(cruises)} 条航次 / {len(pts):,} 点 -> "
              f"{n_px:,} 个 15″ 像素 ({n_px/ship.size*100:.2f}%)")
    if n_px < 50:
        raise RuntimeError("船载像素太少")

    bathy = load_bathy_15s(lon_range, lat_range, lon_hr, lat_hr)
    kf = BathyPriorKF(hr_grid_shape=(ny_hr, nx_hr), lr_grid_shape=(ny_lr, nx_lr),
                      region_size_deg=lat_range[1] - lat_range[0], alpha=ALPHA)
    g15 = kf.build(gravity_lr=glr, ship_15s=ship, train_mask=train_mask,
                   lon_hr=lon_hr, lat_hr=lat_hr, lon_range=lon_range,
                   lat_range=lat_range, upscale=UPSCALE, bathy_15s=bathy,
                   n_iter=N_ITER, smooth_sigma=SMOOTH_SIGMA)
    return g15, lon_hr, lat_hr, dict(n_cruises=len(cruises), n_pts=len(pts), n_px=n_px)


def region_slices(lon_hr, lat_hr):
    """把区域 15″ 栅格映射到全球 15″ 网格的整数切片（要求边界是 15″ 的整数倍）。"""
    gi0 = int(round((lat_hr[0] - (LAT0 + 0.5 / 240)) * 240))
    gj0 = int(round((lon_hr[0] - (LON0 + 0.5 / 240)) * 240))
    gi1, gj1 = gi0 + lat_hr.size, gj0 + lon_hr.size
    exp_lat = LAT0 + (np.arange(gi0, gi1) + 0.5) / 240
    exp_lon = LON0 + (np.arange(gj0, gj1) + 0.5) / 240
    assert np.allclose(exp_lat, lat_hr, atol=1e-9), "纬度方向与全球网格未对齐"
    assert np.allclose(exp_lon, lon_hr, atol=1e-9), "经度方向与全球网格未对齐"
    assert 0 <= gi0 and gi1 <= NLAT and 0 <= gj0 and gj1 <= NLON, "区域越出全球网格"
    return slice(gi0, gi1), slice(gj0, gj1)


def stamp_provenance(ds):
    """把代码版本写入产品属性, 让文件自证是由哪一版代码生成的。"""
    import hashlib
    import subprocess

    repo = Path(__file__).parent.parent

    def git(*args):
        try:
            r = subprocess.run(["git", "-C", str(repo), *args],
                               capture_output=True, encoding="utf-8", errors="replace")
            return (r.stdout or "").strip()
        except Exception:
            return ""

    src_dir = Path(__file__).parent
    srcs = sorted(f.name for f in src_dir.glob("*.py"))
    h = hashlib.sha256()
    for f in srcs:
        h.update((src_dir / f).read_bytes())
    ds.code_commit = git("rev-parse", "HEAD") or "unknown"
    ds.code_commit_subject = git("log", "-1", "--format=%s") or "unknown"
    ds.code_worktree_clean = "yes" if not git("status", "--porcelain") else "NO"
    ds.code_src_sha256 = h.hexdigest()
    ds.code_src_files = f"{len(srcs)} files: " + ",".join(srcs)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="*", default=list(DEFAULT_REGIONS))
    ap.add_argument("--out", default="kalman_gravity_15s_v2.nc")
    ap.add_argument("--band", type=int, default=480, help="每带纬度行数(15 arc-sec)")
    a = ap.parse_args()

    print("=" * 78)
    print("  全球 15 arc-sec 重力场产品 v2（分级: 上采样基底 + 船载区 KF 解算）")
    print("=" * 78)
    t_all = time.time()

    # ---------- 1. 区域解算 ----------
    print("\n[1/3] 区域 15 arc-sec KF 解算")
    solved = []
    for key in a.regions:
        if key not in REGIONS:
            print(f"    跳过未知区域 {key}")
            continue
        t0 = time.time()
        try:
            g15, lon_hr, lat_hr, diag = solve_region(REGIONS[key])
            si, sj = region_slices(lon_hr, lat_hr)
            solved.append(dict(key=key, label=REGIONS[key]["label"], g=g15,
                               si=si, sj=sj, diag=diag))
            print(f"      -> {REGIONS[key]['label']} 完成 {time.time()-t0:.0f}s, "
                  f"切片 lat[{si.start}:{si.stop}] lon[{sj.start}:{sj.stop}]")
        except Exception as e:
            import traceback
            print(f"      !! {key} 失败: {type(e).__name__}: {e}")
            traceback.print_exc()

    # ---------- 2. 打开输出文件 ----------
    out_path = OUT / a.out
    print(f"\n[2/3] 写出 {out_path}")
    ds_out = nc.Dataset(str(out_path), "w", format="NETCDF4")
    ds_out.createDimension("lat", NLAT)
    ds_out.createDimension("lon", NLON)
    v_lat = ds_out.createVariable("lat", "f8", ("lat",))
    v_lon = ds_out.createVariable("lon", "f8", ("lon",))
    v_g = ds_out.createVariable("gravity_anomaly", "f4", ("lat", "lon"),
                                zlib=True, complevel=4)
    v_s = ds_out.createVariable("source", "i1", ("lat", "lon"),
                                zlib=True, complevel=4)
    v_g.units = "mGal"
    v_g.long_name = "Free-air gravity anomaly (15 arc-sec)"
    v_s.long_name = "0 = upsampled 1-arcmin base, 1 = regional KF 15-arcsec solution"
    lat_v = LAT0 + (np.arange(NLAT) + 0.5) / 240.0
    lon_v = LON0 + (np.arange(NLON) + 0.5) / 240.0
    v_lat[:] = lat_v
    v_lon[:] = lon_v

    # ---------- 3. 逐带写出基底 + 覆盖区域解 ----------
    print("\n[3/3] 逐纬度带写出（基底 = SSV 1' 的 4x 双线性上采样, 像元中心对齐）")
    ds_ssv = xr.open_dataset(str(DATA_DIR / "ssv32" / "grav_33.1.nc"), engine=NC_ENGINE)
    ssv_z = ds_ssv["z"]
    if ssv_z.shape[0] * UPSCALE != NLAT or ssv_z.shape[1] * UPSCALE != NLON:
        raise RuntimeError(
            f"SSV 维度 {ssv_z.shape} 与全球 15 arc-sec 网格 {NLAT}x{NLON} 不成 4 倍关系")
    # 全球 15 arc-sec 网格 = SSV 1' 网格的像元中心 4x 细化:
    #   1' 第 j 行覆盖 15 arc-sec 的第 4j..4j+3 行, 是精确的整数关系, 不需要坐标插值。
    band = a.band - (a.band % UPSCALE)
    for i0 in range(0, NLAT, band):
        i1 = min(i0 + band, NLAT)
        j0 = i0 // UPSCALE
        j1 = (i1 + UPSCALE - 1) // UPSCALE
        tile = ssv_z.isel(lat=slice(j0, j1)).values.astype(np.float64)
        big = zoom(tile, (UPSCALE, UPSCALE), order=1)
        first = i0 - UPSCALE * j0                     # 恒在 0..3
        g_band = big[first:first + (i1 - i0), :NLON]
        s_band = np.zeros(g_band.shape, dtype=np.int8)
        if g_band.shape != (i1 - i0, NLON):
            raise RuntimeError(f"带对齐失败: {g_band.shape} vs {(i1-i0, NLON)}")
        for r in solved:
            ia, ib = r["si"].start, r["si"].stop
            lo = max(ia, i0)
            hi = min(ib, i1)
            if hi <= lo:
                continue
            g_band[lo - i0:hi - i0, r["sj"]] = r["g"][lo - ia:hi - ia, :]
            s_band[lo - i0:hi - i0, r["sj"]] = 1
        v_g[i0:i1, :] = g_band.astype(np.float32)
        v_s[i0:i1, :] = s_band
        if (i0 // band) % 10 == 0:
            print(f"    lat 行 {i0:>6}-{i1:<6} ({lat_v[i0]:+.2f}~{lat_v[i1-1]:+.2f}) "
                  f"KF 覆盖 {int(s_band.sum())/s_band.size*100:5.1f}%")
    ds_ssv.close()

    ds_out.generator = "src/run_global_v2.py"
    ds_out.created = time.strftime("%Y-%m-%d %H:%M:%S")
    ds_out.grid_registration = "pixel (cell-centred, node_offset=1)"
    ds_out.base_field = "SSV33.1 1-arcmin, 4x bilinear upsample (cell-centred)"
    ds_out.regional_method = (f"BathyPriorKF: block-filled SSV + alpha*bathy_highpass, "
                              f"alpha={ALPHA}, n_iter={N_ITER}, sigma={SMOOTH_SIGMA}px, "
                              f"Q=4, R_sat=9, R_ship=1 mGal^2")
    ds_out.regions_solved = ",".join(r["label"] for r in solved)
    ds_out.note = ("Graded product: outside the shipborne-covered regions the field is "
                   "the 1-arcmin SSV field upsampled 4x and carries NO new 15-arcsec "
                   "information; use the `source` variable to distinguish. "
                   "The old kalman_gravity_15s.nc is superseded (it used a gridline "
                   "coordinate convention and pasted in residual-gravity patches "
                   "without datum recovery, giving -13..-28 mGal regional biases).")
    stamp_provenance(ds_out)
    ds_out.close()

    size = out_path.stat().st_size
    print(f"\n完成: {out_path}  ({size/2**30:.2f} GiB)  总用时 {time.time()-t_all:.0f}s")
    print(f"  解算区域: {', '.join(r['label'] for r in solved)}")


if __name__ == "__main__":
    main()
