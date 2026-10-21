"""解析 JODC MGD77 定长记录，提取日本窗口内的船测重力。"""
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\gravity\kalman_gravity")
ZIP = ROOT / "data" / "ship_gravity" / "jodc"
OUT = ROOT / "data" / "ship_gravity" / "jodc_gravity.npz"
LON, LAT = (141.0, 149.0), (32.0, 40.0)


def parse_line(line):
    try:
        lat = float(line[27:35]) / 1e5
        lon = float(line[35:44]) / 1e5
        depth = float(line[72:78]) / 10
        grav = float(line[79:84]) / 10
    except ValueError:
        return None
    if abs(grav) >= 900 or abs(grav) > 500:
        grav = np.nan
    if depth <= -99999 or depth >= 999999:
        depth = np.nan
    return lat, lon, depth, grav


def main():
    lats, lons, gravs, deps, voyages = [], [], [], [], []
    for z in sorted(ZIP.glob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            names = [n for n in zf.namelist() if n.endswith("-data")]
            for n in names:
                with zf.open(n) as fh:
                    for raw in fh:
                        line = raw.decode("latin-1").rstrip("\r\n")
                        if len(line) < 84:
                            continue
                        v = parse_line(line)
                        if v is None:
                            continue
                        la, lo, de, gr = v
                        if not (LAT[0] <= la <= LAT[1] and LON[0] <= lo <= LON[1]):
                            continue
                        lats.append(la); lons.append(lo)
                        gravs.append(gr); deps.append(de)
                        voyages.append(z.stem)
    lats = np.asarray(lats); lons = np.asarray(lons)
    gravs = np.asarray(gravs); deps = np.asarray(deps)
    ok = np.isfinite(gravs)
    print("points in bbox", len(lats), "with gravity", ok.sum())
    print("voyages with gravity", len(set(np.asarray(voyages)[ok])))
    print("grav range", np.nanmin(gravs), np.nanmax(gravs),
          "median", np.nanmedian(gravs))
    np.savez_compressed(OUT, lat=lats, lon=lons, grav=gravs, depth=deps,
                        voyage=np.asarray(voyages))
    print("saved", OUT)


if __name__ == "__main__":
    main()
