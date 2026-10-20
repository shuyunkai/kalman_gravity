"""多尺度分解与断裂属性（位场经典算子）。

实现均为频域/空域经典公式：
  - 向上延拓      F[g]·exp(-k h)                 （位场延拓，Blakely 1995）
  - 垂向导数      F^-1[ |k|·F[g] ]               （与 utils.compute_vgg 同源）
  - 水平总梯度    sqrt(gx^2 + gy^2)              （THG, Cordell & Grauch 1985）
  - tilt angle    atan2(gz, THG)                 （Miller & Singh 1994）
  - theta map     acos(THG / sqrt(THG^2+gz^2))    （Wijns et al. 2005）
  - 多尺度        不同延拓高度之差 = 不同深度带的场（Worms 等多尺度边缘分析思路）
断裂识别用 tilt/THG 的脊线（局部极大 + 分位数阈值）作为线性构造候选。
"""
from __future__ import annotations

import numpy as np


def wavenumbers(nx: int, ny: int, dx: float, dy: float):
    """返回 (K, kx, ky)，单位 rad/m。dx/dy 为米。"""
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    return np.sqrt(KX ** 2 + KY ** 2), KX, KY


def upward_continuation(field: np.ndarray, dx: float, dy: float, height: float) -> np.ndarray:
    """向上延拓 height 米（height>0 向上）。"""
    K, _, _ = wavenumbers(field.shape[1], field.shape[0], dx, dy)
    return np.real(np.fft.ifft2(np.fft.fft2(field) * np.exp(-K * height)))


def vertical_derivative(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """垂向导数（|k| 滤波），单位 mGal/m。"""
    K, _, _ = wavenumbers(field.shape[1], field.shape[0], dx, dy)
    K = K.copy()
    K[0, 0] = 0.0
    return np.real(np.fft.ifft2(np.fft.fft2(field) * K))


def horizontal_gradients(field: np.ndarray, dx: float, dy: float):
    gy, gx = np.gradient(field, dy, dx)
    return gx, gy


def thg(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """水平总梯度。"""
    gx, gy = horizontal_gradients(field, dx, dy)
    return np.hypot(gx, gy)


def tilt_angle(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """tilt = atan2(gz, THG)，单位弧度；零值附近对应地质体边缘。"""
    gz = vertical_derivative(field, dx, dy)
    return np.arctan2(gz, thg(field, dx, dy))


def theta_map(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """theta = acos(THG / sqrt(THG^2 + gz^2))，边缘处趋近 0。"""
    gz = vertical_derivative(field, dx, dy)
    t = thg(field, dx, dy)
    return np.arccos(np.clip(t / np.sqrt(t ** 2 + gz ** 2 + 1e-12), -1, 1))


def bandpass(field: np.ndarray, dx: float, dy: float,
             lam_min: float, lam_max: float, order: int = 4) -> np.ndarray:
    """带通：保留 lam_min < 波长 < lam_max（米）。"""
    K, _, _ = wavenumbers(field.shape[1], field.shape[0], dx, dy)
    K = K.copy()
    K[0, 0] = 1e-12
    lp = 1.0 / (1.0 + (K * lam_min / (2 * np.pi)) ** (2 * order))
    hp = 1.0 - 1.0 / (1.0 + (K * lam_max / (2 * np.pi)) ** (2 * order))
    return np.real(np.fft.ifft2(np.fft.fft2(field) * lp * hp))


def multi_scale(field: np.ndarray, dx: float, dy: float,
                heights=(0.0, 5e3, 10e3, 20e3)):
    """按延拓高度做尺度分离，返回 {高度: 场} 与相邻高度差（深度带）。"""
    cont = {h: upward_continuation(field, dx, dy, h) for h in heights}
    bands = {}
    hs = list(heights)
    for a, b in zip(hs[:-1], hs[1:]):
        bands[(a, b)] = cont[a] - cont[b]
    return cont, bands


def ridge_mask(attr: np.ndarray, size: int = 7, pct: float = 88.0) -> np.ndarray:
    """属性场脊线：局部极大且高于分位数阈值，用作线性构造候选。"""
    from scipy.ndimage import maximum_filter

    v = np.nan_to_num(attr, nan=0.0)
    thr = np.percentile(np.abs(v), pct)
    peak = v == maximum_filter(v, size=size)
    return peak & (np.abs(v) >= thr)


def zhang_suen_thin(mask: np.ndarray) -> np.ndarray:
    """Zhang–Suen 细化，把粗边缘压成单像元线（线性构造提取用）。"""
    img = mask.astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            P2 = p[:-2, 1:-1]; P3 = p[:-2, 2:]; P4 = p[1:-1, 2:]
            P5 = p[2:, 2:]; P6 = p[2:, 1:-1]; P7 = p[2:, :-2]
            P8 = p[1:-1, :-2]; P9 = p[:-2, :-2]
            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
            seq = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8)
                    for i in range(8))
            if step == 0:
                cond = ((img == 1) & (B >= 2) & (B <= 6) & (A == 1)
                        & (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0))
            else:
                cond = ((img == 1) & (B >= 2) & (B <= 6) & (A == 1)
                        & (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0))
            if cond.any():
                img[cond] = 0
                changed = True
    return img.astype(bool)


def lineament_mask(field: np.ndarray, dx: float, dy: float,
                   pct: float = 88.0) -> np.ndarray:
    """带通场 → THG 阈值 → 细化，得到单像元线性构造候选。"""
    t = thg(field, dx, dy)
    thr = np.percentile(t, pct)
    return zhang_suen_thin(t >= thr)


def filter_lineaments(mask: np.ndarray, min_area: int = 60,
                     min_elong: float = 2.0) -> np.ndarray:
    """去掉小斑块与近圆形闭合环，只保留细长线性体（断裂候选）。"""
    from scipy.ndimage import label, find_objects

    lab, _ = label(mask, structure=np.ones((3, 3), dtype=int))
    keep = np.zeros_like(mask, dtype=bool)
    for i, sl in enumerate(find_objects(lab), start=1):
        if sl is None:
            continue
        comp = lab[sl] == i
        if int(comp.sum()) < min_area:
            continue
        h, w = comp.shape
        elong = max(h, w) / max(1, min(h, w))
        if elong < min_elong:
            continue
        keep[sl] |= comp
    return keep
