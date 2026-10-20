"""反演结果的分频评估、振幅标定与剖面。"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from bathymetry_inversion import bandpass


def band_metrics(h_inv, h_true, dx, dy,
                 bands=((20e3, 50e3), (50e3, 100e3), (100e3, 200e3))):
    """分频段比较反演与真值：RMSE、相关、振幅比。"""
    out = []
    for lo, hi in bands:
        a = bandpass(h_inv, dx, dy, lo, hi)
        b = bandpass(h_true, dx, dy, lo, hi)
        m = np.isfinite(a) & np.isfinite(b)
        rmse = float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))
        corr = float(np.corrcoef(a[m], b[m])[0, 1])
        amp = float(np.std(a[m]) / np.std(b[m]))
        out.append(dict(band=f"{lo/1e3:.0f}–{hi/1e3:.0f} km", rmse=rmse,
                        corr=corr, amp=amp))
    return out


def amplitude_slope(h_inv, h_true):
    """真值 = a·反演 + b 的最小二乘斜率（振幅标定系数）。"""
    m = np.isfinite(h_inv) & np.isfinite(h_true)
    a, b = np.polyfit(h_inv[m], h_true[m], 1)
    return float(a), float(b)


def profile(h_inv, h_true, lon, lat, p0, p1, n=400):
    """沿 p0→p1 的剖面（经度, 纬度），返回 (dist_km, h_inv, h_true)。"""
    f_inv = RegularGridInterpolator((lat, lon), h_inv, bounds_error=False,
                                    fill_value=np.nan)
    f_true = RegularGridInterpolator((lat, lon), h_true, bounds_error=False,
                                     fill_value=np.nan)
    lo = np.linspace(p0[0], p1[0], n)
    la = np.linspace(p0[1], p1[1], n)
    hi = f_inv(np.column_stack([la, lo]))
    ht = f_true(np.column_stack([la, lo]))
    dx_km = (lo - lo[0]) * 111.32 * np.cos(np.radians(np.mean(la)))
    dy_km = (la - la[0]) * 111.32
    dist = np.hypot(dx_km, dy_km)
    return dist, hi, ht
