"""频域卡尔曼/维纳反演：把地形谱系数当状态，用先验谱与观测噪声做逐波数增益。"""
from __future__ import annotations

import math

import numpy as np

from bathymetry_inversion import _K, parker_forward, lowpass, G, MGAL


def spectral_kf_inversion(dg, dx, dy, rho, z0, prior_rms=2500.0, slope=2.5,
                          noise_mgal=2.0, n_iter=8, lam_min=25e3,
                          relax=0.8, taper=0.05, R_spec=None):
    ny, nx = dg.shape
    wy = np.ones(ny); wx = np.ones(nx)
    n_y = max(1, int(ny * taper)); n_x = max(1, int(nx * taper))
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_y)))
    wy[:n_y] = ramp; wy[-n_y:] = ramp[::-1]
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_x)))
    wx[:n_x] = ramp; wx[-n_x:] = ramp[::-1]
    dg_w = dg * np.outer(wy, wx)

    K = _K(nx, ny, dx, dy)
    A = 2 * np.pi * G * rho * np.exp(-K * z0) * MGAL   # mGal / m
    k0 = 2 * np.pi / 100e3
    P0 = prior_rms ** 2 * (k0 / np.maximum(K, 1e-9)) ** slope
    R = noise_mgal ** 2 if R_spec is None else R_spec
    gain = P0 * A / (A ** 2 * P0 + R)

    h = np.zeros((ny, nx))
    hist = []
    for it in range(n_iter):
        pred = parker_forward(h, dx, dy, rho, z0, n_terms=4)
        resid = dg_w - pred
        h_corr = np.real(np.fft.ifft2(np.fft.fft2(resid) * gain))
        h_corr = lowpass(h_corr, dx, dy, lam_min, order=6)
        h = h + relax * h_corr
        hist.append(float(np.sqrt(np.mean(resid ** 2))))
        print(f"  iter {it+1}: residual {hist[-1]:.2f} mGal")
    return h, hist
