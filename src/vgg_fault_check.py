"""SSV 垂直重力梯度 VGG 独立做 GM9 断裂：不走 GEBCO 先验。"""
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))

from scale_decomposition import bandpass, lineament_mask, filter_lineaments
from fault_validation import densify, fault_match

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real" / "three_parts"
LON, LAT = (170.3, 172.1), (-46.4, -45.0)


def main():
    ds = xr.open_dataset(ROOT / "data" / "ssv32" / "curv_33.1.nc")
    lon = ds["lon"].values
    lat = ds["lat"].values
    z = ds["z"].values
    ds.close()
    if lon.max() > 180:
        lon = (lon + 180) % 360 - 180
        order = np.argsort(lon)
        lon = lon[order]
        z = z[:, order]
    i0 = np.searchsorted(lon, LON[0]); i1 = np.searchsorted(lon, LON[1])
    j0 = np.searchsorted(lat, LAT[0]); j1 = np.searchsorted(lat, LAT[1])
    z = z[j0:j1, i0:i1]
    lon = lon[i0:i1]; lat = lat[j0:j1]
    dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = float(lat[1] - lat[0]) * 111320
    z = np.nan_to_num(z, nan=0.0)
    print("grid", z.shape, "range", z.min(), z.max())

    lines = []
    for f in json.loads((OUT / "nz_gm9_offshore.json").read_text(encoding="utf-8"))["features"]:
        geo = f.get("geometry") or {}
        if geo.get("type") != "LineString":
            continue
        a = np.asarray(geo["coordinates"], float)
        if ((a[:, 0] >= LON[0]) & (a[:, 0] <= LON[1]) &
                (a[:, 1] >= LAT[0]) & (a[:, 1] <= LAT[1])).any():
            lines.append(a)
    idx = np.arange(len(lines)); np.random.default_rng(7).shuffle(idx)
    tx, ty = densify([lines[i] for i in idx[len(lines) // 2:]], 0.5)

    b = bandpass(z, dx, dy, 5e3, 20e3)
    m = filter_lineaments(lineament_mask(b, dx, dy, pct=60), 3, 1.2)
    for r in (5, 10, 20):
        st = fault_match(m, tx, ty, lon, lat, radius_km=r)
        print(f"VGG r={r}: cover {st['recall']*100:.1f}% "
              f"prec {st['precision']*100:.1f}% px {int(m.sum())}")


if __name__ == "__main__":
    main()
