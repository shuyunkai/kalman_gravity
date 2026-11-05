"""莫霍面反演：常用重力场(SSV) vs 本项目 KF 场，标准 Parker–Oldenburg。"""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

sys.path.insert(0, str(Path(__file__).parent))

from bathymetry_inversion import bandpass, parker_forward, parker_inversion
from kf_novel import BathyPriorKF

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real" / "three_parts"
LON = (141.0, 149.0)
LAT = (32.0, 40.0)
RHO = 400.0
BAND = (200e3, 500e3)


def crust_moho():
    b = np.loadtxt(ROOT / "data" / "crust1" / "crust1.bnds")
    m = -b[:, 8].reshape(180, 360)          # 正深度 km
    lat = np.arange(89.5, -90, -1)
    lon = np.arange(-179.5, 180, 1)
    iy = (lat <= LAT[1]) & (lat >= LAT[0])
    ix = (lon >= LON[0]) & (lon <= LON[1])
    return m[np.ix_(iy, ix)], lat[iy], lon[ix]


def main():
    d = np.load(OUT / "kf_inputs.npz")
    lon = d["lon_hr"]; lat = d["lat_hr"]
    ny, nx = d["bathy_15s"].shape
    dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = float(lat[1] - lat[0]) * 111320

    # KF 场
    kf = BathyPriorKF((ny, nx), d["gravity_lr"].shape, (8.0, 8.0),
                      ship_noise_std=10.0, alpha=0.5)
    g_kf = kf.build(d["gravity_lr"].astype(float), d["ship_grav"],
                    d["train_mask"], lon, lat, LON, LAT, upscale=4,
                    bathy_15s=d["bathy_15s"], n_iter=6, smooth_sigma=4.0)
    g_ssv = zoom(d["gravity_lr"], (4, 4), order=1)

    # CRUST1.0 真值，上采样到 15″
    true_km, clat, clon = crust_moho()
    true_hr = zoom(true_km, (ny / true_km.shape[0], nx / true_km.shape[1]), order=1)
    z0 = float(np.mean(true_hr)) * 1000.0
    # 平滑初值：500 km 低通（用高斯近似）
    sig_px = 500e3 / (2 * np.pi) / dx
    smooth = gaussian_filter(true_hr, sigma=sig_px)
    h0 = (z0 / 1000.0 - smooth) * 1000.0     # 正向上，m

    print(f"z0={z0/1000:.2f} km, h0 std={h0.std()/1000:.2f} km")
    g0 = bandpass(parker_forward(h0, dx, dy, RHO, z0, n_terms=2), dx, dy, *BAND)

    def inv(g):
        gb = bandpass(g, dx, dy, *BAND)
        res = gb - g0
        print("   gb/g0/res std", float(gb.std()), float(g0.std()), float(res.std()))
        out = parker_inversion(res, dx, dy, RHO, z0, n_iter=6,
                               lam_min=BAND[0], relax=0.2, verbose=False)
        dh = out[0] if isinstance(out, tuple) else out
        print("   dh std km", float(dh.std()) / 1000.0)
        moho = z0 / 1000.0 - (h0 + dh) / 1000.0
        return moho

    m0 = z0 / 1000.0 - h0 / 1000.0
    for name, g in (("SSV", g_ssv), ("KF", g_kf)):
        m = inv(g)
        # 1° 单元采样
        sampled = np.zeros_like(true_km)
        for i, la in enumerate(clat):
            for j, lo in enumerate(clon):
                iy = np.argmin(np.abs(lat - la)); ix = np.argmin(np.abs(lon - lo))
                sampled[i, j] = m[iy, ix]
        rmse = float(np.sqrt(np.mean((sampled - true_km) ** 2)))
        corr = float(np.corrcoef(sampled.ravel(), true_km.ravel())[0, 1])
        print(f"{name}: RMSE {rmse:.2f} km, corr {corr:.2f}, "
              f"range {m.min():.1f}..{m.max():.1f}")
    # 初值误差
    s0 = np.zeros_like(true_km)
    for i, la in enumerate(clat):
        for j, lo in enumerate(clon):
            iy = np.argmin(np.abs(lat - la)); ix = np.argmin(np.abs(lon - lo))
            s0[i, j] = m0[iy, ix]
    print(f"initial smooth: RMSE {np.sqrt(np.mean((s0-true_km)**2)):.2f} km")


if __name__ == "__main__":
    main()
