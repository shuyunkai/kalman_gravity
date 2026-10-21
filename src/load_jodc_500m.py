"""把 JODC 500m 水深瓦片拼成日本窗口 15″ 网格。"""
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\gravity\kalman_gravity")
SRC = ROOT / "data" / "jodc_bathy"
OUT = SRC / "jodc_500m_15s.npz"
LON, LAT = (141.0, 149.0), (32.0, 40.0)
RES = 1.0 / 240.0


def main():
    lats, lons, deps = [], [], []
    for z in sorted(SRC.glob("mesh500_*.zip")):
        try:
            zf = zipfile.ZipFile(z)
        except zipfile.BadZipFile:
            continue
        with zf:
            for n in zf.namelist():
                if not n.endswith(".txt"):
                    continue
                with zf.open(n) as fh:
                    for raw in fh:
                        p = raw.decode("ascii", "ignore").split()
                        if len(p) < 4:
                            continue
                        try:
                            la, lo, de = float(p[1]), float(p[2]), float(p[3])
                        except ValueError:
                            continue
                        if LAT[0] <= la <= LAT[1] and LON[0] <= lo <= LON[1]:
                            lats.append(la); lons.append(lo); deps.append(de)
    lats = np.asarray(lats); lons = np.asarray(lons); deps = np.asarray(deps)
    print("points", len(deps), "depth range", deps.min(), deps.max())

    nx = int(round((LON[1] - LON[0]) / RES)) + 1
    ny = int(round((LAT[1] - LAT[0]) / RES)) + 1
    ix = np.clip(((lons - LON[0]) / RES).round().astype(int), 0, nx - 1)
    iy = np.clip(((lats - LAT[0]) / RES).round().astype(int), 0, ny - 1)
    idx = iy * nx + ix
    cnt = np.bincount(idx, minlength=ny * nx).reshape(ny, nx)
    sums = np.bincount(idx, weights=deps, minlength=ny * nx).reshape(ny, nx)
    grid = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)
    lon = LON[0] + np.arange(nx) * RES
    lat = LAT[0] + np.arange(ny) * RES
    np.savez_compressed(OUT, depth=grid, lon=lon, lat=lat)
    print("grid", grid.shape, "coverage", np.isfinite(grid).mean())
    print("saved", OUT)


if __name__ == "__main__":
    main()
