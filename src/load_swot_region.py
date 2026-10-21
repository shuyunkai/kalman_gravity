"""把 SWOT L2 LR SSH 裁到日本窗口并缓存为 npz。"""
import glob
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(r"D:\gravity\kalman_gravity")
LON, LAT = (141.0, 149.0), (32.0, 40.0)


def main():
    files = sorted(glob.glob(str(ROOT / "data" / "swot" / "*Basic*.nc")))
    if not files:
        print("no SWOT Basic files")
        return
    lats, lons, sshs, sshas, uncs, times = [], [], [], [], [], []
    for f in files:
        ds = xr.open_dataset(f)
        lat = ds.latitude.values; lon = ds.longitude.values
        q = ds.ssh_karin_qual.values
        m = (q == 0) & np.isfinite(ds.ssh_karin.values)
        m &= (lon >= LON[0]) & (lon <= LON[1]) & (lat >= LAT[0]) & (lat <= LAT[1])
        if not m.any():
            ds.close(); continue
        lats.append(lat[m]); lons.append(lon[m])
        sshs.append(ds.ssh_karin.values[m])
        sshas.append(ds.ssha_karin.values[m])
        uncs.append(ds.ssh_karin_uncert.values[m])
        n = m.shape[0]
        t = np.repeat(ds.time.values[:, None], m.shape[1], axis=1)[m]
        times.append(t.astype("datetime64[s]").astype("int64"))
        ds.close()
    if not lats:
        print("no valid pixels in bbox")
        return
    out = dict(
        lat=np.concatenate(lats), lon=np.concatenate(lons),
        ssh=np.concatenate(sshs), ssha=np.concatenate(sshas),
        unc=np.concatenate(uncs), time=np.concatenate(times),
    )
    p = ROOT / "data" / "swot" / "swot_japan_subset.npz"
    np.savez_compressed(p, **out)
    print("pixels", len(out["lat"]), "->", p)
    print("lon", out["lon"].min(), out["lon"].max(),
          "lat", out["lat"].min(), out["lat"].max())
    print("ssh m", np.percentile(out["ssh"], [1, 50, 99]))
    print("ssha m", np.percentile(out["ssha"], [1, 50, 99]))
    print("unc mm", np.median(out["unc"]) * 1000)


if __name__ == "__main__":
    main()
