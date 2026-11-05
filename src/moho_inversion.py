"""莫霍面反演原型：KF 15″ 场 50–500 km → Parker → 与 CRUST1.0 比较。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from bathymetry_inversion import bandpass, parker_inversion
from kf_novel import BathyPriorKF

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real" / "three_parts"
LON = (141.0, 149.0)
LAT = (32.0, 40.0)


def load_crust_moho():
    b = np.loadtxt(ROOT / "data" / "crust1" / "crust1.bnds")
    m = b[:, 8].reshape(180, 360)
    lat = np.arange(89.5, -90, -1)
    lon = np.arange(-179.5, 180, 1)
    iy = (lat <= LAT[1]) & (lat >= LAT[0])
    ix = (lon >= LON[0]) & (lon <= LON[1])
    return -m[np.ix_(iy, ix)], lat[iy], lon[ix]


def main():
    d = np.load(OUT / "kf_inputs.npz")
    lon = d["lon_hr"]; lat = d["lat_hr"]
    ny, nx = d["bathy_15s"].shape
    dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = float(lat[1] - lat[0]) * 111320

    kf = BathyPriorKF((ny, nx), d["gravity_lr"].shape, (8.0, 8.0),
                      ship_noise_std=10.0, alpha=0.5)
    g = kf.build(d["gravity_lr"].astype(float), d["ship_grav"],
                 d["train_mask"], lon, lat, LON, LAT, upscale=4,
                 bathy_15s=d["bathy_15s"], n_iter=6, smooth_sigma=4.0)

    gb = bandpass(g, dx, dy, 200e3, 500e3)
    z0 = 15000.0
    rho = 400.0
    out = parker_inversion(gb, dx, dy, rho, z0, n_iter=10,
                           lam_min=200e3, relax=0.3, verbose=True)
    h = out[0] if isinstance(out, tuple) else out
    moho = z0 - h

    true, clat, clon = load_crust_moho()
    print("CRUST moho km", true.min(), true.max(), true.mean())

    # 1° 单元中心采样
    inv = np.zeros_like(true)
    for i, la in enumerate(clat):
        for j, lo in enumerate(clon):
            iy = np.argmin(np.abs(lat - la))
            ix = np.argmin(np.abs(lon - lo))
            inv[i, j] = moho[iy, ix] / 1000.0
    err = inv - true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    corr = float(np.corrcoef(inv.ravel(), true.ravel())[0, 1])
    print(f"Moho inversion: RMSE {rmse:.2f} km, corr {corr:.2f}, "
          f"inv range {inv.min():.1f}..{inv.max():.1f}")

    fig, axes = plt.subplots(1, 4, figsize=(14.6, 3.8), dpi=220)
    ext = [lon[0], lon[-1], lat[0], lat[-1]]
    v = np.percentile(np.abs(gb), 99)
    axes[0].imshow(gb, cmap="RdBu_r", origin="lower", vmin=-v, vmax=v,
                   extent=ext, aspect="auto")
    axes[0].set_title("(a) 重力 50–500 km", fontsize=9)
    im = axes[1].imshow(moho / 1000.0, cmap="viridis_r", origin="lower",
                        extent=ext, aspect="auto")
    axes[1].set_title("(b) 反演莫霍面深度 (km)", fontsize=9)
    fig.colorbar(im, ax=axes[1], shrink=0.8, pad=0.02)
    axes[2].scatter(true.ravel(), inv.ravel(), s=12, c="#1F3864")
    lim = [true.min() - 1, true.max() + 1]
    axes[2].plot(lim, lim, "r--", lw=1)
    axes[2].set_xlim(lim); axes[2].set_ylim(lim)
    axes[2].set_xlabel("CRUST1.0 (km)"); axes[2].set_ylabel("反演 (km)")
    axes[2].set_title(f"(c) RMSE {rmse:.1f} km, corr {corr:.2f}", fontsize=9)
    axes[2].grid(alpha=0.3)
    i = np.argmin(np.abs(lat - 36.0))
    axes[3].plot(clat, true[:, np.argmin(np.abs(clon - 145.0))], "k-o",
                 ms=3, label="CRUST1.0")
    axes[3].plot(lat[::80], (moho[i, ::80] / 1000.0), color="#C0392B",
                 label="反演")
    axes[3].invert_xaxis()
    axes[3].set_xlabel("纬度 (°N)"); axes[3].set_ylabel("莫霍面深度 (km)")
    axes[3].set_title("(d) 经向剖面 145°E", fontsize=9)
    axes[3].legend(fontsize=6, frameon=False); axes[3].grid(alpha=0.3)
    fig.tight_layout()
    p = OUT / "fig_moho_validation.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    main()
