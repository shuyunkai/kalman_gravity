"""汇总 JODC MGD77 netCDF 的 Free-air 异常到日本窗口 npz。"""
import glob
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(r"D:\gravity\kalman_gravity")
OUT = ROOT / "data" / "ship_gravity" / "jodc_fa.npz"
LON, LAT = (141.0, 149.0), (32.0, 40.0)


def main():
    lat, lon, fa, voy = [], [], [], []
    for f in glob.glob(str(ROOT / "data" / "ship_gravity" / "jodc_nc" / "*.nc")):
        ds = xr.open_dataset(f)
        la = ds.latitude.values; lo = ds.longitude.values
        fr = ds.Free.values
        m = np.isfinite(fr) & (lo >= LON[0]) & (lo <= LON[1]) \
            & (la >= LAT[0]) & (la <= LAT[1])
        if m.any():
            lat.append(la[m]); lon.append(lo[m]); fa.append(fr[m])
            voy += [Path(f).stem] * int(m.sum())
        ds.close()
    if not lat:
        print("no data")
        return
    lat = np.concatenate(lat); lon = np.concatenate(lon); fa = np.concatenate(fa)
    np.savez_compressed(OUT, lat=lat, lon=lon, fa=fa,
                        voyage=np.asarray(voy))
    print("points", len(fa), "voyages", len(set(voy)),
          "range", np.percentile(fa, [1, 50, 99]))
    print("saved", OUT)


if __name__ == "__main__":
    main()
