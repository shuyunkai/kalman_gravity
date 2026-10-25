"""Te 反演：重力—地形导纳 + 薄板挠曲模型，KF vs SSV。"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from scale_decomposition import bandpass

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real" / "three_parts" / "newdeck"
G = 6.6743e-11
E = 1.0e11          # Young's modulus Pa
NU = 0.25
RHO = 1640.0         # load/infill density contrast kg/m3
G0 = 9.81
DEPTH = 3000.0      # mean water depth m
TE_GRID = np.array([1, 2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 80], float)
NAVY, PALE, RED = "#0B2545", "#D6E2F0", "#C0392B"
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
    K = np.hypot(KX, KY); K[0, 0] = 1e-12
    return K


def theoretical(k, te_m):
    D = E * te_m ** 3 / (12 * (1 - NU ** 2))
    return (2 * np.pi * G * RHO * np.exp(-k * DEPTH)
            / (1 + D * k ** 4 / (RHO * G0)))


def observed_admittance(g, bathy, dx, dy, nb=30):
    gb = bandpass(g, dx, dy, 50e3, 200e3)
    hb = bandpass(bathy, dx, dy, 50e3, 200e3)
    K = k_grid(g.shape, dx, dy)
    Gg = np.fft.fft2(gb); Hh = np.fft.fft2(hb)
    Sgh = Gg * np.conj(Hh); Shh = np.abs(Hh) ** 2
    k = K[K > 0]
    bins = np.linspace(2 * np.pi / 200e3, 2 * np.pi / 50e3, nb)
    idx = np.digitize(K.ravel(), bins)
    z = np.zeros(nb); kk = np.zeros(nb)
    for i in range(1, nb):
        m = idx == i
        if m.sum() < 10:
            continue
        z[i] = np.real(Sgh.ravel()[m].sum() / max(Shh.ravel()[m].sum(), 1e-30))
        kk[i] = np.mean(K.ravel()[m])
    ok = (z > 0) & (kk > 0)
    return kk[ok], z[ok]


def fit_te(g, bathy, dx, dy):
    k, z = observed_admittance(g, bathy, dx, dy)
    zm = z * 1e-5  # mGal/m -> s^-2
    best = None
    for te in TE_GRID:
        model = theoretical(k, te * 1000.0)
        misfit = float(np.mean((np.log10(zm) - np.log10(model)) ** 2))
        if best is None or misfit < best[0]:
            best = (misfit, te)
    return best[1], best[0], k, zm


def main():
    lon, lat, ssv, kf, bathy = load()
    dx = (lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = (lat[1] - lat[0]) * 111320
    te1, m1, k1, z1 = fit_te(ssv, bathy, dx, dy)
    te2, m2, k2, z2 = fit_te(kf, bathy, dx, dy)
    print("SSV Te", te1, "misfit", m1)
    print("KF  Te", te2, "misfit", m2)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.0), dpi=220)
    ax = axes[0]
    ax.loglog(1 / k1 * 2 * np.pi / 1e3, z1 * 1e5, "o", color=PALE, ms=4, label="1′ 观测")
    ax.loglog(1 / k2 * 2 * np.pi / 1e3, z2 * 1e5, "o", color=NAVY, ms=3, label="15″ 观测")
    kk = np.logspace(np.log10(k1.min()), np.log10(k1.max()), 100)
    ax.loglog(2 * np.pi / kk / 1e3, theoretical(kk, te1 * 1000) * 1e5,
              "--", color=RED, lw=1.2, label=f"1′ 模型 Te={te1:.0f} km")
    ax.loglog(2 * np.pi / kk / 1e3, theoretical(kk, te2 * 1000) * 1e5,
              "-", color=NAVY, lw=1.2, label=f"15″ 模型 Te={te2:.0f} km")
    ax.set_xlabel("波长 (km)"); ax.set_ylabel("导纳 (mGal/km)")
    ax.set_title("(a) 观测导纳与理论导纳", fontsize=10)
    ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.25, which="both")

    ax = axes[1]
    misfit1 = [np.mean((np.log10(z1) - np.log10(theoretical(k1, te * 1000))) ** 2)
               for te in TE_GRID]
    misfit2 = [np.mean((np.log10(z2) - np.log10(theoretical(k2, te * 1000))) ** 2)
               for te in TE_GRID]
    ax.semilogy(TE_GRID, misfit1, "o-", color=PALE, label="1′")
    ax.semilogy(TE_GRID, misfit2, "o-", color=NAVY, label="15″")
    ax.axvline(te1, color=PALE, ls=":"); ax.axvline(te2, color=NAVY, ls=":")
    ax.set_xlabel("Te (km)"); ax.set_ylabel("对数导纳残差")
    ax.set_title("(b) Te 拟合残差", fontsize=10)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_te_compare.png", bbox_inches="tight")
    print("saved", OUT / "fig_te_compare.png")


if __name__ == "__main__":
    main()
