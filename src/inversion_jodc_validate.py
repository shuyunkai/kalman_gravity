"""用 JODC 500m 水深独立验证 KF 反演（20–200 km）。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from bathymetry_inversion import bandpass as bp
from kf_inversion import spectral_kf_inversion

ROOT = Path(__file__).parent.parent
FIELD = ROOT / "output_real" / "test_field_15s_jodc.npz"
JODC = ROOT / "data" / "jodc_bathy" / "jodc_500m_15s.npz"


def main():
    f = np.load(FIELD)
    g = f["g15"].astype(float); bathy = f["bathy"].astype(float)
    lon, lat = f["lon"], f["lat"]
    dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = float(lat[1] - lat[0]) * 111320
    z0 = float(-np.median(bathy))
    dg = bp(g, dx, dy, 20e3, 200e3)
    h, _ = spectral_kf_inversion(dg, dx, dy, 2200.0, z0, prior_rms=2500.0,
                                 n_iter=8, lam_min=25e3, relax=0.6)

    j = np.load(JODC)
    depth = j["depth"][:h.shape[0], :h.shape[1]]
    target = z0 - depth
    m = np.isfinite(target)
    print("valid cells", m.sum(), "z0", round(z0, 1))

    def stats(a, b, mask):
        d = a[mask] - b[mask]
        return (float(np.sqrt(np.mean(d ** 2))),
                float(np.corrcoef(a[mask], b[mask])[0, 1]))

    rmse, corr = stats(h, target, m)
    print(f"full 20-200: RMSE {rmse:.1f} m corr {corr:.3f}")
    for lo, hi in ((20e3, 50e3), (50e3, 100e3), (100e3, 200e3)):
        ha = bp(h, dx, dy, lo, hi)
        ta = bp(target, dx, dy, lo, hi)
        r, c = stats(ha, ta, m)
        print(f"{lo/1e3:.0f}-{hi/1e3:.0f} km: RMSE {r:.1f} m corr {c:.3f}")


if __name__ == "__main__":
    main()
