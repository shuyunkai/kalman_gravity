"""集合 KF：扰动观测生成 20 个成员，得到非均匀的集合方差，再检验其作 R(k) 的效果。"""
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
    g0 = d["gravity_lr"].astype(float)
    ship = d["ship_grav"].astype(float)
    mask = d["train_mask"]
    dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = float(lat[1] - lat[0]) * 111320
    z0 = float(np.mean(-bathy))
    h = z0 + bathy

    rng = np.random.default_rng(42)
    members = []
    for m in range(20):
        sat = g0 + rng.normal(0, 3.0, g0.shape)
        sh = ship.copy()
        sh[mask] = ship[mask] + rng.normal(0, 1.0, mask.sum())
        kf = BathyPriorKF(d["ssv_hr"].shape, g0.shape, (8, 8), alpha=0.5)
        members.append(kf.build(sat, sh, mask, lon, lat, (141.0, 149.0),
                                (32.0, 40.0), upscale=4, bathy_15s=bathy,
                                n_iter=6, smooth_sigma=4.0))
        print("member", m + 1, "done")
    ens = np.stack(members)
    mean = ens.mean(axis=0)
    var = ens.var(axis=0)
    print(f"ensemble spread: mean={var.mean():.2f}, p10={np.percentile(var,10):.2f}, "
          f"p90={np.percentile(var,90):.2f}, max={var.max():.1f}")
    np.savez_compressed(OUT / "ensemble_kf.npz", mean=mean, var=var)

    dg = bandpass(mean, dx, dy, 20e3, 200e3)
    ht = bandpass(h, dx, dy, 20e3, 200e3)
    N = var.size
    Fp = np.fft.fft2(np.sqrt(np.maximum(var, 0)))
    Rk = np.abs(Fp) ** 2 / N

    for tag, spec in (("constant", None), ("ensemble R(k)", Rk)):
        hk, _ = spectral_kf_inversion(dg, dx, dy, 2200.0, z0, prior_rms=2500.0,
                                      noise_mgal=2.0, n_iter=8, lam_min=25e3,
                                      relax=0.6, R_spec=spec)
        hb = bandpass(hk, dx, dy, 20e3, 200e3)
        print(f"{tag}: RMSE={np.sqrt(np.mean((hb-ht)**2)):.1f} m, "
              f"corr={np.corrcoef(hb.ravel(), ht.ravel())[0,1]:.3f}")


if __name__ == "__main__":
    main()
