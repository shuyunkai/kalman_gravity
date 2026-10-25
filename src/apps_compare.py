"""五项应用的应用适配性对比：只替换重力场。"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter

sys.path.insert(0, str(Path(__file__).parent))

from scale_decomposition import bandpass

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real" / "three_parts" / "newdeck"
NAVY, PALE, RED = "#0B2545", "#D6E2F0", "#C0392B"
G = 6.6743e-11
RHO = 2670.0
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False})


def load():
    d = np.load(ROOT / "output_real/three_parts/ship_regions/G_Japan.npz")
    lon, lat = d["lon"], d["lat"]
    iy = (lat >= 32) & (lat <= 40); ix = (lon >= 141) & (lon <= 149)
    return (lon[ix], lat[iy], d["ssv"][np.ix_(iy, ix)].astype(float),
            d["kf"][np.ix_(iy, ix)].astype(float),
            d["bathy"][np.ix_(iy, ix)].astype(float))


def k_grid(shape, dx, dy):
    kx = 2 * np.pi * np.fft.fftfreq(shape[1], dx)
    ky = 2 * np.pi * np.fft.fftfreq(shape[0], dy)
    KX, KY = np.meshgrid(kx, ky)
    K = np.hypot(KX, KY)
    K[0, 0] = 1e-12
    return K


def metrics(g, bathy, dx, dy):
    out = {}
    gb = bandpass(g, dx, dy, 5e3, 50e3)
    gy, gx = np.gradient(gb, dy, dx)
    grad = np.hypot(gx, gy) * 1e-5          # mGal/m -> (m/s^2)/m? keep mGal/m
    grad = np.hypot(gx, gy)                 # mGal/m
    ocean = bathy < 0
    out["nav_error_m"] = float(np.median(0.5 / np.maximum(grad[ocean], 1e-9)))

    g20 = bandpass(g, dx, dy, 20e3, 200e3)
    K = k_grid(g.shape, dx, dy)
    mask = (K > 2 * np.pi / 200e3) & (K < 2 * np.pi / 20e3)
    Ks = np.where(mask, K, np.inf)
    N = np.real(np.fft.ifft2(np.fft.fft2(g20) * 1e-5
                             / (2 * G * RHO * Ks)))
    N = bandpass(N, dx, dy, 20e3, 200e3)
    out["geoid_rms_m"] = float(np.std(N[ocean]))

    hb = bandpass(bathy, dx, dy, 50e3, 100e3)
    gb2 = bandpass(g, dx, dy, 50e3, 100e3)
    Sgh_sum = 0.0; Sgg_sum = 0.0; Shh_sum = 0.0
    step = max(256, g.shape[0] // 4)
    for i0 in range(0, g.shape[0] - step + 1, step):
        for j0 in range(0, g.shape[1] - step + 1, step):
            Gh = np.fft.fft2(hb[i0:i0+step, j0:j0+step])
            Gg = np.fft.fft2(gb2[i0:i0+step, j0:j0+step])
            Sgh_sum = Sgh_sum + Gg * np.conj(Gh)
            Sgg_sum = Sgg_sum + np.abs(Gg) ** 2
            Shh_sum = Shh_sum + np.abs(Gh) ** 2
    Kt = k_grid((step, step), dx, dy)
    bt = (Kt > 2*np.pi/100e3) & (Kt < 2*np.pi/50e3)
    coh = np.abs(Sgh_sum) ** 2 / np.maximum(Sgg_sum * Shh_sum, 1e-30)
    out["te_coherence"] = float(np.mean(coh[bt]))

    f = bandpass(g, dx, dy, 20e3, 50e3)
    mx = maximum_filter(f, size=15)
    out["feature_density"] = float(np.sum((f == mx) & (f > np.percentile(f, 90)) & ocean))

    K2 = k_grid(g.shape, dx, dy)
    up = np.real(np.fft.ifft2(np.fft.fft2(g20) * np.exp(-K2 * 500e3)))
    out["orbit_acc_mgal"] = float(np.std(up[ocean]))
    return out


def main():
    lon, lat, ssv, kf, bathy = load()
    dx = (lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = (lat[1] - lat[0]) * 111320
    m1 = metrics(ssv, bathy, dx, dy)
    m2 = metrics(kf, bathy, dx, dy)
    print(json.dumps({"SSV": m1, "KF": m2}, ensure_ascii=False, indent=1))
    (OUT / "apps_metrics.json").write_text(
        json.dumps({"SSV": m1, "KF": m2}, ensure_ascii=False, indent=1),
        encoding="utf-8")

    labels = ["导航误差\n(m)", "Te 相干", "特征密度"]
    keys = ["nav_error_m", "te_coherence", "feature_density"]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), dpi=220)
    for ax, lab, key in zip(axes, labels, keys):
        vals = [m1[key], m2[key]]
        ax.bar([0, 1], vals, color=[PALE, NAVY], width=0.6)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["1′", "15″"], fontsize=8)
        ax.set_title(lab, fontsize=9)
        for i, v in enumerate(vals):
            ax.text(i, v * 1.02, f"{v:.3g}", ha="center", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("同一公式、只替换重力场：三项应用指标对比", fontsize=10, color=RED)
    fig.tight_layout()
    fig.savefig(OUT / "fig_apps_compare.png", bbox_inches="tight")
    print("saved", OUT / "fig_apps_compare.png")


if __name__ == "__main__":
    main()
