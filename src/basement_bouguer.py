"""标准基底反演：水层+沉积改正 → Parker，比较 SSV/KF/BOL。"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent))

from bathymetry_inversion import bandpass, parker_forward, parker_inversion

ROOT = Path(__file__).parent.parent
SRC = ROOT / "output_real" / "three_parts" / "ship_regions"
GLOB = ROOT / "data" / "GlobSed" / "GlobSed_package3" / "GlobSed-v3.nc"
BAND = (10e3, 50e3)
RHO_W, RHO_S, RHO_B = 1030.0, 2200.0, 2700.0
G = 6.6743e-11
K_W = 2 * np.pi * G * (RHO_S - RHO_W) * 1e5   # mGal/m
K_S = 2 * np.pi * G * (RHO_B - RHO_S) * 1e5   # mGal/m

REGIONS = {
    "D_Nankai": dict(lon=(124.0, 132.0), lat=(24.0, 31.0)),
    "F_PhilSea": dict(lon=(139.0, 152.0), lat=(13.0, 26.0)),
    "G_Japan": dict(lon=(132.0, 152.0), lat=(26.0, 42.0)),
}


def load_globsed(lon_rng, lat_rng):
    ds = xr.open_dataset(GLOB)
    lon = ds["lon"].values; lat = ds["lat"].values; z = ds["z"].values
    ds.close()
    iy = (lat >= lat_rng[0]) & (lat <= lat_rng[1])
    ix = (lon >= lon_rng[0]) & (lon <= lon_rng[1])
    z = np.nan_to_num(z[np.ix_(iy, ix)].astype(float), nan=0.0)
    return lat[iy], lon[ix], z


def main():
    for name, cfg in REGIONS.items():
        print("===", name)
        d = np.load(SRC / f"{name}.npz")
        lon, lat, bathy = d["lon"], d["lat"], d["bathy"].astype(float)
        ny, nx = bathy.shape
        dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
        dy = float(lat[1] - lat[0]) * 111320

        glat, glon, ts = load_globsed(cfg["lon"], cfg["lat"])
        t0g = gaussian_filter(ts, sigma=300e3 / (111320 * 0.1))
        Y, X = np.meshgrid(lat, lon, indexing="ij")
        pts = np.c_[Y.ravel(), X.ravel()]
        ts_hr = RegularGridInterpolator((glat, glon), ts, bounds_error=False,
                                        fill_value=0.0)(pts).reshape(ny, nx)
        t0_hr = RegularGridInterpolator((glat, glon), t0g, bounds_error=False,
                                        fill_value=0.0)(pts).reshape(ny, nx)

        ocean = bathy < 0
        dw = np.where(ocean, -bathy, 0.0)
        ts_hr = np.where(ocean, ts_hr, 0.0)
        t0_hr = np.where(ocean, t0_hr, 0.0)

        d0 = dw + t0_hr
        z0 = float(np.mean(d0[ocean]))
        h0 = z0 - d0
        g0 = bandpass(parker_forward(h0, dx, dy, RHO_B - RHO_S, z0, n_terms=2),
                      dx, dy, *BAND)

        def invert(g):
            gb = g + K_W * dw + K_S * t0_hr
            gb = bandpass(gb, dx, dy, *BAND)
            out = parker_inversion(gb - g0, dx, dy, RHO_B - RHO_S, z0,
                                   n_iter=4, lam_min=30e3, relax=0.05,
                                   verbose=False)
            dh = out[0] if isinstance(out, tuple) else out
            dd = z0 - (h0 + dh)
            return np.clip(dd - dw, 0.0, 20000.0)

        def score(t):
            a = bandpass(t, dx, dy, *BAND)
            b = bandpass(ts_hr, dx, dy, *BAND)
            m = ocean & np.isfinite(a) & np.isfinite(b)
            return (float(np.sqrt(np.mean((a[m] - b[m]) ** 2))),
                    float(np.corrcoef(a[m], b[m])[0, 1]))

        print("  initial", score(t0_hr))
        for tag in ("ssv", "kf", "bol"):
            print(f"  {tag}", score(invert(d[tag])))


if __name__ == "__main__":
    main()
