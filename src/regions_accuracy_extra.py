"""补齐三区三种常用 15″ 场的留出 RMSE：双线性、频域融合、LSC。"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent))

from three_method_compare import load_region_data
from freq_fusion_baseline import freq_fusion_reconstruct

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"

REGIONS = {
    "D_Nankai": dict(lon=(124.0, 132.0), lat=(24.0, 31.0),
                     val="YK14-14_gravity_dmo"),
    "H_Kuroshio": dict(lon=(134.0, 141.0), lat=(31.0, 36.0),
                       val="YK18-14_gravity_dmo"),
    "G_Japan": dict(lon=(132.0, 152.0), lat=(26.0, 42.0),
                    val="MR22-04_gravity_dmo"),
}


def rmse_at(model, lon, lat, pts):
    ix = np.clip(np.searchsorted(lon, pts[:, 0]), 0, len(lon) - 1)
    iy = np.clip(np.searchsorted(lat, pts[:, 1]), 0, len(lat) - 1)
    d = model[iy, ix] - pts[:, 2]
    return float(np.sqrt(np.mean(d ** 2)))


def lsc_field(ssv, ship, mask, dx_km, sigma_km=40.0):
    """LSC/最优插值：SSV 为背景，船测残差按高斯协方差做归一化卷积。"""
    m = mask & np.isfinite(ship)
    res = np.where(m, ship - ssv, 0.0)
    sig = sigma_km / max(dx_km, 1e-6)
    num = gaussian_filter(res, sig, mode="nearest")
    den = gaussian_filter(m.astype(float), sig, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        return ssv + np.where(den > 1e-6, num / den, 0.0)


def main():
    res = json.loads((OUT / "regions_coherence.json").read_text(encoding="utf-8"))
    for name, cfg in REGIONS.items():
        print("===", name)
        d = load_region_data(cfg["lon"], cfg["lat"], val_cruise=cfg["val"], upscale=4)
        lon, lat = d["lon_hr"], d["lat_hr"]
        pts = np.asarray(d["val_points"], float)
        dx_km = (lon[1] - lon[0]) * 111.0 * np.cos(np.radians(lat.mean()))
        bil = rmse_at(d["ssv_hr"], lon, lat, pts)
        freq = freq_fusion_reconstruct(d["ssv_hr"], d["ship_grav"], d["train_mask"],
                                       lon, lat, cfg["lon"], cfg["lat"])
        f_rmse = rmse_at(freq, lon, lat, pts)
        lsc = lsc_field(d["ssv_hr"], d["ship_grav"], d["train_mask"], dx_km)
        l_rmse = rmse_at(lsc, lon, lat, pts)
        res[name].update(bilinear=bil, freq=f_rmse, lsc=l_rmse)
        print(f"  bilinear {bil:.4f} freq {f_rmse:.4f} lsc {l_rmse:.4f} "
              f"ckf {res[name]['ckf']:.4f}")
    (OUT / "regions_coherence.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
