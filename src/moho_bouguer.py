"""莫霍面反演：布格改正 + CRUST1.0 初值 + Parker，比较 SSV/KF/BOL。"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent))

from bathymetry_inversion import bandpass, parker_forward, parker_inversion

ROOT = Path(__file__).parent.parent
SRC = ROOT / "output_real" / "three_parts" / "ship_regions" / "G_Japan.npz"
CRUST = ROOT / "data" / "crust1" / "crust1.bnds"
GLOB = ROOT / "data" / "GlobSed" / "GlobSed_package3" / "GlobSed-v3.nc"
LON, LAT = (141.0, 149.0), (32.0, 40.0)
BAND = (50e3, 500e3)
RHO = 400.0
RHO_W, RHO_S, RHO_B = 1030.0, 2200.0, 2700.0
G = 6.6743e-11
K_W = 2 * np.pi * G * (RHO_S - RHO_W) * 1e5
K_S = 2 * np.pi * G * (RHO_B - RHO_S) * 1e5


def crust_moho():
    b = np.loadtxt(CRUST)
    m = -b[:, 8].reshape(180, 360)
    lat = np.arange(89.5, -90, -1)
    lon = np.arange(-179.5, 180, 1)
    iy = (lat <= LAT[1]) & (lat >= LAT[0])
    ix = (lon >= LON[0]) & (lon <= LON[1])
    return m[np.ix_(iy, ix)], lat[iy], lon[ix]


def globsed():
    ds = xr.open_dataset(GLOB)
    lon = ds["lon"].values; lat = ds["lat"].values; z = ds["z"].values
    ds.close()
    iy = (lat >= LAT[0]) & (lat <= LAT[1])
    ix = (lon >= LON[0]) & (lon <= LON[1])
    z = np.nan_to_num(z[np.ix_(iy, ix)].astype(float), nan=0.0)
    return lat[iy], lon[ix], z


def main():
    d = np.load(SRC)
    lon, lat, bathy = d["lon"], d["lat"], d["bathy"].astype(float)
    iy = (lat >= LAT[0]) & (lat <= LAT[1])
    ix = (lon >= LON[0]) & (lon <= LON[1])
    lon = lon[ix]; lat = lat[iy]
    bathy = bathy[np.ix_(iy, ix)]
    ny, nx = bathy.shape
    dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = float(lat[1] - lat[0]) * 111320

    true_km, clat, clon = crust_moho()
    true_hr = RegularGridInterpolator((clat, clon), true_km,
                                      bounds_error=False, fill_value=np.nan)(
        np.c_[np.repeat(lat, nx), np.tile(lon, ny)]).reshape(ny, nx)
    true_hr = np.nan_to_num(true_hr, nan=float(np.nanmean(true_km)))
    z0 = float(np.mean(true_hr)) * 1000.0
    d0 = gaussian_filter(true_hr, sigma=500e3 / (2 * np.pi) / dx)
    h0 = (z0 / 1000.0 - d0) * 1000.0

    glat, glon, ts = globsed()
    t0 = gaussian_filter(ts, sigma=300e3 / (111320 * 0.1))
    t0_hr = RegularGridInterpolator((glat, glon), t0, bounds_error=False,
                                    fill_value=0.0)(
        np.c_[np.repeat(lat, nx), np.tile(lon, ny)]).reshape(ny, nx)
    ocean = bathy < 0
    dw = np.where(ocean, -bathy, 0.0)
    t0_hr = np.where(ocean, t0_hr, 0.0)

    g0 = bandpass(parker_forward(h0, dx, dy, RHO, z0, n_terms=2),
                  dx, dy, *BAND)

    def invert(g):
        gb = g + K_W * dw + K_S * t0_hr
        gb = bandpass(gb, dx, dy, *BAND)
        out = parker_inversion(gb - g0, dx, dy, RHO, z0, n_iter=4,
                               lam_min=150e3, relax=0.05, verbose=False)
        dh = out[0] if isinstance(out, tuple) else out
        return z0 / 1000.0 - (h0 + dh) / 1000.0

    def score(m):
        m1 = gaussian_filter(m, sigma=0.5)
        t1 = gaussian_filter(true_km, sigma=0.5)
        return (float(np.sqrt(np.mean((m1 - t1) ** 2))),
                float(np.corrcoef(m1.ravel(), t1.ravel())[0, 1]))

    def sample(m):
        out = np.zeros_like(true_km)
        for i, la in enumerate(clat):
            for j, lo in enumerate(clon):
                iy2 = np.argmin(np.abs(lat - la))
                ix2 = np.argmin(np.abs(lon - lo))
                out[i, j] = m[iy2, ix2]
        return out

    print("initial", score(sample(d0)))
    for tag in ("ssv", "kf", "bol"):
        f = d[tag][np.ix_(iy, ix)].astype(float)
        m = invert(f)
        # 1° 采样
        samp = np.zeros_like(true_km)
        for i, la in enumerate(clat):
            for j, lo in enumerate(clon):
                iy2 = np.argmin(np.abs(lat - la)); ix2 = np.argmin(np.abs(lon - lo))
                samp[i, j] = m[iy2, ix2]
        print(tag, score(samp))


if __name__ == "__main__":
    main()
