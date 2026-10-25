"""结构引导（各向异性）KF 原型：预测步保留脊线细节。"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom as sp_zoom

sys.path.insert(0, str(Path(__file__).parent))

from run_pipeline import load_ssv_region
from three_method_compare import load_bathy_15s
from utils import cell_centers
from kf_novel import BathyPriorKF, block_fill
from kalman_gravity import KalmanGravityReconstructor
from scale_decomposition import bandpass, lineament_mask, filter_lineaments
from fault_validation import densify, fault_match

LON, LAT = (170.3, 172.1), (-46.4, -45.0)


def build(g_lr, bathy, lon, lat, w_keep=0.7, p0=2.0, n_iter=6):
    ny, nx = bathy.shape
    up = 4
    kf = BathyPriorKF((ny, nx), g_lr.shape, (1.4, 1.8), process_noise_std=p0,
                      alpha=0.5, p0_std=p0, q_std=p0, adaptive_ship=False)
    ssv_block = block_fill(g_lr, up)
    boug = kf._bathy_to_bouguer(bathy, lon, lat)
    hp = kf._high_pass(boug, lon, lat)
    prior = ssv_block + 0.5 * hp

    gy, gx = np.gradient(hp)
    Jxx = gaussian_filter(gx * gx, 8)
    Jyy = gaussian_filter(gy * gy, 8)
    Jxy = gaussian_filter(gx * gy, 8)
    coh = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) / (Jxx + Jyy + 1e-12)
    w = np.clip(coh, 0, 1) ** 2

    r = KalmanGravityReconstructor(grid_shape=(ny, nx),
                                   region_size_deg=(LAT[1] - LAT[0]),
                                   lr_grid_shape=g_lr.shape)
    r._initialized = True
    r.x = prior.ravel().copy()
    r.P_diag = p0 ** 2 * np.ones(r.x.size)
    r.Q_diag = p0 ** 2 * np.ones(r.x.size)
    r.R_sat_val = 9.0
    r.R_ship_val = 1.0
    r.gravity_lr = g_lr
    r.gravity_ship = np.full((ny, nx), np.nan)
    r.ship_train_mask = np.zeros((ny, nx), dtype=bool)

    def predict(self, smooth_sigma=4.0):
        x2 = self.x.reshape(self.ny, self.nx)
        xs = gaussian_filter(x2, sigma=smooth_sigma)
        x2 = xs + w_keep * w * (x2 - xs)
        self.x = x2.ravel()
        factor = 1.0 + smooth_sigma * 0.1
        self.P_diag = self.P_diag * factor + self.Q_diag
        cap = 100.0 * (np.median(self.P_diag) + 1e-10)
        self.P_diag = np.minimum(self.P_diag, cap)

    import types
    r.predict = types.MethodType(predict, r)
    return r.run(n_iterations=n_iter, smooth_sigma=4.0, verbose=False)


def main():
    g_lr, _, _, _ = load_ssv_region(LON, LAT)
    ny, nx = g_lr.shape; up = 4
    lon = cell_centers(LON[0], LON[1], nx * up)
    lat = cell_centers(LAT[0], LAT[1], ny * up)
    bathy = load_bathy_15s(LON, LAT, lon, lat)
    dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = float(lat[1] - lat[0]) * 111320

    lines = []
    for f in json.load(open(r"output_real/three_parts/nz_gm9_offshore.json", encoding="utf-8"))["features"]:
        geo = f.get("geometry") or {}
        if geo.get("type") != "LineString":
            continue
        a = np.asarray(geo["coordinates"], float)
        if ((a[:, 0] >= LON[0]) & (a[:, 0] <= LON[1]) &
                (a[:, 1] >= LAT[0]) & (a[:, 1] <= LAT[1])).any():
            lines.append(a)
    idx = np.arange(len(lines)); np.random.default_rng(7).shuffle(idx)
    tx, ty = densify([lines[i] for i in idx[len(lines) // 2:]], 0.5)

    for wk in (0.0, 0.4, 0.7, 1.0):
        g = build(g_lr, bathy, lon, lat, w_keep=wk)
        b = bandpass(g, dx, dy, 5e3, 20e3)
        m = filter_lineaments(lineament_mask(b, dx, dy, pct=70), 20, 1.8)
        st = fault_match(m, tx, ty, lon, lat, radius_km=5)
        print(f"w={wk}: 5km {st['recall']*100:.1f}%  "
              f"prec {st['precision']*100:.1f}%  px {int(m.sum())}")


if __name__ == "__main__":
    main()
