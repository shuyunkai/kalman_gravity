"""用 KF 重建的逐像元方差构造 R(k)，检验能否改善反演。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from kf_novel import BathyPriorKF
from bathymetry_inversion import bandpass
from kf_inversion import spectral_kf_inversion

OUT = Path(__file__).parent.parent / "output_real" / "three_parts"


def main():
    d = np.load(OUT / "kf_inputs.npz")
    lon, lat = d["lon_hr"], d["lat_hr"]
    bathy = d["bathy_15s"].astype(float)
    dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = float(lat[1] - lat[0]) * 111320
    z0 = float(np.mean(-bathy))
    h = z0 + bathy

    kf = BathyPriorKF(d["ssv_hr"].shape, d["gravity_lr"].shape, (8, 8), alpha=0.5)
    g15 = kf.build(d["gravity_lr"].astype(float), d["ship_grav"], d["train_mask"],
                   lon, lat, (141.0, 149.0), (32.0, 40.0), upscale=4,
                   bathy_15s=bathy, n_iter=6, smooth_sigma=4.0)
    P = kf.P_diag.reshape(g15.shape)
    print(f"P: mean={P.mean():.2f}, median={np.median(P):.2f}, "
          f"p10={np.percentile(P,10):.2f}, p90={np.percentile(P,90):.2f} mGal^2")

    # R(k)：把不确定度场 sqrt(P) 的功率谱作为观测噪声谱
    N = P.size
    Fp = np.fft.fft2(np.sqrt(P))
    Rk = np.abs(Fp) ** 2 / N
    print(f"Rk: mean={Rk.mean():.2f}, max={Rk.max():.2f} mGal^2")

    dg = bandpass(g15, dx, dy, 20e3, 200e3)
    ht = bandpass(h, dx, dy, 20e3, 200e3)

    for tag, spec in (("constant R", None), ("KF R(k)", Rk)):
        hk, _ = spectral_kf_inversion(dg, dx, dy, 2200.0, z0, prior_rms=2500.0,
                                      noise_mgal=2.0, n_iter=8, lam_min=25e3,
                                      relax=0.6, R_spec=spec)
        hb = bandpass(hk, dx, dy, 20e3, 200e3)
        print(f"{tag}: RMSE={np.sqrt(np.mean((hb-ht)**2)):.1f} m, "
              f"corr={np.corrcoef(hb.ravel(), ht.ravel())[0,1]:.3f}")


if __name__ == "__main__":
    main()
