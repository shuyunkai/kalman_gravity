"""Parker–Oldenburg 重力反演海底地形（频域迭代）。

经典公式：
  Parker (1973):  Δg(k) = 2πGρ e^{-k z0} Σ_{n=1}^{N} (-k)^{n-1}/n! · F[h^n]
  Oldenburg (1974): 以观测 Δg 与正演 Δg 之差反投影，迭代修正 h

参考实现：pyParkO（Eurrreka/pyParkO, Computers & Geosciences 2026），
本模块只依赖 numpy，参数命名与其 wh/alpha 滤波思路一致。

约定：
  z0 : 平均水深（正，单位 m）
  h  : 相对 z0 的地形起伏（正向上，单位 m），即 h = -bathy - z0
  Δg : 重力异常，单位 mGal
"""
from __future__ import annotations

import math

import numpy as np

G = 6.6743e-11          # m^3 kg^-1 s^-2
MGAL = 1e5              # m/s^2 -> mGal


def _K(nx, ny, dx, dy):
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K = np.hypot(KX, KY)
    K[0, 0] = 1e-12
    return K


def lowpass(field, dx, dy, lam_min, order=4):
    """保留波长 > lam_min 的分量（Butterworth 低通）。"""
    K = _K(field.shape[1], field.shape[0], dx, dy)
    H = 1.0 / (1.0 + (K * lam_min / (2 * np.pi)) ** (2 * order))
    return np.real(np.fft.ifft2(np.fft.fft2(field) * H))


def bandpass(field, dx, dy, lam_min, lam_max, order=4):
    K = _K(field.shape[1], field.shape[0], dx, dy)
    lp = 1.0 / (1.0 + (K * lam_min / (2 * np.pi)) ** (2 * order))
    hp = 1.0 - 1.0 / (1.0 + (K * lam_max / (2 * np.pi)) ** (2 * order))
    return np.real(np.fft.ifft2(np.fft.fft2(field) * lp * hp))


def parker_forward(h, dx, dy, rho, z0, n_terms=5, mu=0.0):
    """由界面起伏 h 正演重力异常（mGal）。

    mu > 0 为变密度（VDM）：密度差随深度衰减，核函数由 (-k) 变为 -(k+mu)，
    对应 pyParkO 中 (frequency - mu) 的写法（符号随 h 正方向约定）。
    """
    K = _K(h.shape[1], h.shape[0], dx, dy)
    Kmu = K + mu
    F = np.fft.fft2(h.astype(np.float64))
    total = np.zeros_like(F)
    hn = np.ones_like(h, dtype=np.float64)
    for n in range(1, n_terms + 1):
        hn = hn * h
        total += ((-Kmu) ** (n - 1)) / math.factorial(n) * np.fft.fft2(hn)
    dg = 2 * np.pi * G * rho * np.exp(-K * z0) * total
    return np.real(np.fft.ifft2(dg)) * MGAL


def parker_inversion(dg, dx, dy, rho, z0, n_iter=10, lam_min=20e3,
                     relax=0.8, taper=0.05, verbose=True, mu=0.0):
    """由重力异常反演界面起伏 h（m），返回 h 与迭代诊断。"""
    ny, nx = dg.shape
    # 边缘余弦 taper，抑制 FFT 环绕
    wy = np.ones(ny); wx = np.ones(nx)
    n_y = max(1, int(ny * taper)); n_x = max(1, int(nx * taper))
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_y)))
    wy[:n_y] = ramp; wy[-n_y:] = ramp[::-1]
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_x)))
    wx[:n_x] = ramp; wx[-n_x:] = ramp[::-1]
    dg_w = dg * np.outer(wy, wx)

    K = _K(nx, ny, dx, dy)
    h = np.zeros((ny, nx))
    hist = []
    best_h, best_rmse = h.copy(), np.inf
    for it in range(n_iter):
        pred = parker_forward(h, dx, dy, rho, z0, mu=mu)
        resid = dg_w - pred
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        hist.append(rmse)
        if verbose:
            print(f"  iter {it+1:2d}: residual RMSE = {rmse:.3f} mGal")
        if rmse < best_rmse:
            best_rmse, best_h = rmse, h.copy()
        elif it > 2:
            print("  early stop: residual increased")
            break
        h_corr = np.real(np.fft.ifft2(
            np.fft.fft2(resid) * np.exp(K * z0) / (2 * np.pi * G * rho * MGAL)))
        h_corr = lowpass(h_corr, dx, dy, lam_min, order=6)
        h = h + relax * h_corr
    return best_h, hist
