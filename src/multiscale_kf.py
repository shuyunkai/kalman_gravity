"""多尺度卡尔曼滤波：把状态拆成粗/中/细三个分量，共享观测。

x = x1 + x2 + x3
  x1: 长波（SSV 平滑）  x2: 中波（SSV 残差 + 地形高频）  x3: 细波（初值 0）
预测：各分量用不同 σ 的高斯平滑；协方差对角、按分量独立膨胀。
更新：卫星块均值、船载点观测都作用于 x1+x2+x3，增益按各分量方差占比分配。
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def block_mean(a, u=4):
    ny, nx = a.shape
    return a.reshape(ny // u, u, nx // u, u).mean(axis=(1, 3))


def block_rep(a, u=4):
    return np.repeat(np.repeat(a, u, axis=0), u, axis=1)


class MultiScaleKF:
    def __init__(self, ssv_hr, terrain_hp, sigmas=(16.0, 4.0, 1.0),
                 q=(16.0, 4.0, 1.0), r_sat=9.0, r_ship=1.0, alpha=0.5):
        c = gaussian_filter(ssv_hr, 16.0)
        self.x = [c.copy(), (ssv_hr - c + alpha * terrain_hp).copy(),
                  np.zeros_like(ssv_hr)]
        self.P = [q[i] * np.ones_like(ssv_hr) for i in range(3)]
        self.sigmas, self.q = sigmas, q
        self.r_sat, self.r_ship = r_sat, r_ship

    def predict(self):
        for i, s in enumerate(self.sigmas):
            self.x[i] = gaussian_filter(self.x[i], s)
            self.P[i] = self.P[i] * (1 + 0.1 * s) + self.q[i]

    def update_sat(self, z):
        pred = block_mean(self.x[0] + self.x[1] + self.x[2])
        Ps = [block_mean(P) for P in self.P]
        denom = Ps[0] + Ps[1] + Ps[2] + self.r_sat
        for i in range(3):
            K = Ps[i] / denom
            self.x[i] += block_rep(K * (z - pred))
            self.P[i] = block_rep((1 - K) * Ps[i])

    def update_ship(self, z, mask):
        pred = self.x[0] + self.x[1] + self.x[2]
        denom = self.P[0] + self.P[1] + self.P[2] + self.r_ship
        for i in range(3):
            K = self.P[i] / denom
            self.x[i][mask] += K[mask] * (z - pred)[mask]
            self.P[i] = (1 - K) * self.P[i]

    def run(self, z_sat, ship, mask, n_iter=6):
        for _ in range(n_iter):
            self.predict()
            self.update_sat(z_sat)
            self.update_ship(ship, mask)
        return self.x
