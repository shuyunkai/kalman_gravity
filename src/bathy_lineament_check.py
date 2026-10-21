"""高分辨率 GMRT 地形 vs GM9 断层：验证“短波信息来自地形”这一判断。"""
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))

from scale_decomposition import bandpass, lineament_mask, filter_lineaments
from fault_validation import densify, fault_match

OUT = Path(__file__).parent.parent / "output_real" / "three_parts"
LON, LAT = (170.3, 172.1), (-46.4, -45.0)


def main():
    ds = xr.open_dataset(OUT / "gmrt_gm9_max.grd")
    nx, ny = [int(v) for v in ds["dimension"].values]
    x0, x1 = ds["x_range"].values
    y0, y1 = ds["y_range"].values
    z = np.flipud(ds["z"].values.reshape(ny, nx))
    ds.close()
    lon = np.linspace(x0, x1, nx)
    lat = np.linspace(y0, y1, ny)
    dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = float(lat[1] - lat[0]) * 111320
    from scipy.ndimage import distance_transform_edt
    bad = ~np.isfinite(z)
    if bad.any():
        idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
        z = z[tuple(idx)]
    print("grid", z.shape, "dx m", round(dx), "dy m", round(dy),
          "range", np.nanmin(z), np.nanmax(z))

    d = json.loads((OUT / "nz_gm9_offshore.json").read_text(encoding="utf-8"))
    lines = []
    for f in d["features"]:
        geo = f.get("geometry") or {}
        if geo.get("type") != "LineString":
            continue
        a = np.asarray(geo["coordinates"], float)
        ins = ((a[:, 0] >= LON[0]) & (a[:, 0] <= LON[1]) &
               (a[:, 1] >= LAT[0]) & (a[:, 1] <= LAT[1]))
        if ins.any():
            lines.append(a)
    idx = np.arange(len(lines)); rng = np.random.default_rng(7); rng.shuffle(idx)
    test = [lines[i] for i in idx[len(lines) // 2:]]
    tx, ty = densify(test, 0.5)

    b = bandpass(z, dx, dy, 1e3, 5e3)
    mask = filter_lineaments(lineament_mask(b, dx, dy, pct=70) & (z < 0),
                             min_area=40, min_elong=1.8)
    print("lineaments", mask.sum())
    for r in (0.5, 1, 2, 5):
        st = fault_match(mask, tx, ty, lon, lat, radius_km=r)
        print("r=%.1f km: cover %.1f (rand %.1f) prec %.1f (rand %.1f)" % (
            r, st["recall"]*100, st["recall_random"]*100,
            st["precision"]*100, st["precision_random"]*100))


if __name__ == "__main__":
    main()
