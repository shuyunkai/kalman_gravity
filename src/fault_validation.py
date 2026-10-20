"""断裂候选的定量验证：走向统计、地震点距离、海沟轴距离。

思路：线性体只是“候选”，需要用独立资料检验其构造意义：
  1. 走向玫瑰图 —— 是否出现与海沟/转换断层一致的优选方位；
  2. 地震点距离 —— 地震（USGS 目录）是否落在候选线性体附近；
  3. 海沟轴距离 —— 轴部线性体是否贴合 GEBCO 最深处。
"""
from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
from scipy.ndimage import label, find_objects
from scipy.spatial import cKDTree

KM_PER_DEG = 111.32


def _xy(lon, lat):
    return lon * KM_PER_DEG * np.cos(np.radians(np.mean(lat))), lat * KM_PER_DEG


def lineament_segments(mask, lon, lat, min_area=80, min_elong=2.2):
    """把线性体掩膜拆成线段，返回 [(angle_deg, length_km, cx, cy), ...]。"""
    lab, _ = label(mask, structure=np.ones((3, 3), dtype=int))
    segs = []
    for i, sl in enumerate(find_objects(lab), start=1):
        if sl is None:
            continue
        comp = lab[sl] == i
        area = int(comp.sum())
        if area < min_area:
            continue
        h, w = comp.shape
        if max(h, w) / max(1, min(h, w)) < min_elong:
            continue
        ys, xs = np.nonzero(comp)
        gy = ys + sl[0].start
        gx = xs + sl[1].start
        X, Y = _xy(lon[gx], lat[gy])
        X0, Y0 = X - X.mean(), Y - Y.mean()
        cov = np.cov(np.vstack([X0, Y0]))
        evals, evecs = np.linalg.eigh(cov)
        vx, vy = evecs[:, int(np.argmax(evals))]
        angle = float(np.degrees(np.arctan2(vy, vx)) % 180.0)
        proj = X0 * vx + Y0 * vy
        length = float(proj.max() - proj.min())
        segs.append((angle, length, float(lon[gx].mean()), float(lat[gy].mean())))
    return segs


def rose_hist(segs, bins=12):
    """按长度加权的走向直方图（0–180°）。"""
    edges = np.linspace(0, 180, bins + 1)
    h = np.zeros(bins)
    for angle, length, *_ in segs:
        k = min(bins - 1, int(angle / 180 * bins))
        h[k] += length
    return np.radians((edges[:-1] + edges[1:]) / 2), h


def _mask_points_km(mask, lon, lat):
    ys, xs = np.nonzero(mask)
    X, Y = _xy(lon[xs], lat[ys])
    return np.column_stack([X, Y])


def distance_stats(mask, q_lon, q_lat, lon, lat, radius_km=20.0, n_random=4000,
                   seed=0):
    """查询点到最近线性体的距离统计，并给随机点基线。"""
    pts = _mask_points_km(mask, lon, lat)
    tree = cKDTree(pts)
    qx, qy = _xy(q_lon, q_lat)
    d, _ = tree.query(np.column_stack([qx, qy]))
    rng = np.random.default_rng(seed)
    rx = rng.uniform(lon.min(), lon.max(), n_random)
    ry = rng.uniform(lat.min(), lat.max(), n_random)
    rX, rY = _xy(rx, ry)
    dr, _ = tree.query(np.column_stack([rX, rY]))
    return dict(
        n=int(len(d)),
        median_km=float(np.median(d)),
        p90_km=float(np.percentile(d, 90)),
        frac_in_radius=float(np.mean(d <= radius_km)),
        random_frac=float(np.mean(dr <= radius_km)),
        dist=d,
    )


def load_usgs_csv(path: Path):
    lons, lats, mags = [], [], []
    with open(path, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                lons.append(float(row["longitude"]))
                lats.append(float(row["latitude"]))
                mags.append(float(row["mag"]))
            except (KeyError, ValueError):
                continue
    return np.asarray(lons), np.asarray(lats), np.asarray(mags)


def trench_axis(bathy, lat):
    """每列最深处所在的纬度（海沟轴近似）。"""
    idx = np.argmin(bathy, axis=0)
    return lat[idx]


def trench_distance(mask, bathy, lat, lon, lon_min=142.0, lon_max=145.5,
                    lat_min=33.5):
    """海沟轴上每点到最近线性体的距离（km）。

    只在海沟实际存在的经度/纬度窗内统计，避免把图幅南缘的深海平原误当轴部。
    """
    sel_lon = (lon >= lon_min) & (lon <= lon_max)
    sub = np.where(lat[:, None] >= lat_min, bathy, np.inf)
    idx = np.argmin(sub, axis=0)
    ty = lat[idx][sel_lon]
    tx = lon[sel_lon]
    pts = _mask_points_km(mask, lon, lat)
    tree = cKDTree(pts)
    X, Y = _xy(tx, ty)
    d, _ = tree.query(np.column_stack([X, Y]))
    return dict(median_km=float(np.median(d)), p90_km=float(np.percentile(d, 90)),
                dist=d)
