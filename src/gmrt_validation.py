"""用 GMRT 独立地形产品验证 KF 反演（GMRT 含多波束，独立于 GEBCO）。"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).parent))

from bathymetry_inversion import bandpass
from kf_inversion import spectral_kf_inversion

OUT = Path(__file__).parent.parent / "output_real" / "three_parts"


def load_gmrt(path):
    ds = xr.open_dataset(path)
    nx, ny = [int(v) for v in ds["dimension"].values]
    x0, x1 = ds["x_range"].values
    y0, y1 = ds["y_range"].values
    z = np.flipud(ds["z"].values.reshape(ny, nx))
    lon = np.linspace(x0, x1, nx)
    lat = np.linspace(y1, y0, ny)
    ds.close()
    return z, lon, lat


def main():
    d = np.load(OUT / "kf_inputs.npz")
    lon, lat = d["lon_hr"], d["lat_hr"]
    bathy = d["bathy_15s"].astype(float)
    g15 = np.load(OUT.parent / "test_field_15s.npz")["g15"].astype(float)
    dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = float(lat[1] - lat[0]) * 111320
    z0 = float(np.mean(-bathy))
    h_true = z0 + bathy

    dg = bandpass(g15, dx, dy, 20e3, 200e3)
    h_inv, _ = spectral_kf_inversion(dg, dx, dy, 2200.0, z0, prior_rms=2500.0,
                                     n_iter=8, lam_min=25e3, relax=0.6)
    h_inv = bandpass(h_inv, dx, dy, 20e3, 200e3)

    z, glon, glat = load_gmrt(OUT / "gmrt_region.grd")
    print("GMRT", z.shape, z.min(), z.max(), glon[0], glon[-1], glat[0], glat[-1])
    f = RegularGridInterpolator((glat, glon), z, bounds_error=False,
                                fill_value=np.nan)
    LLON, LLAT = np.meshgrid(lon, lat)
    gm = f(np.column_stack([LLAT.ravel(), LLON.ravel()])).reshape(lat.size, lon.size)
    gm = np.where(np.isfinite(gm), gm, -4000.0)
    h_gmrt = bandpass(z0 + gm, dx, dy, 20e3, 200e3)
    m = np.isfinite(h_gmrt)
    print(f"vs GMRT: RMSE={np.sqrt(np.mean((h_inv[m]-h_gmrt[m])**2)):.0f} m, "
          f"corr={np.corrcoef(h_inv[m], h_gmrt[m])[0,1]:.3f}")
    hm = bandpass(h_true, dx, dy, 20e3, 200e3)
    print(f"GEBCO vs GMRT: RMSE={np.sqrt(np.mean((hm[m]-h_gmrt[m])**2)):.0f} m, "
          f"corr={np.corrcoef(hm[m], h_gmrt[m])[0,1]:.3f}")


if __name__ == "__main__":
    main()



