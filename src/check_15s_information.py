import sys
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, r"D:\gravity\kalman_gravity\src")
from scale_decomposition import bandpass

d = np.load(r"D:\gravity\kalman_gravity\output_real\test_field_15s.npz")
g15 = d["g15"].astype(float); bathy = d["bathy"].astype(float)
lon, lat = d["lon"], d["lat"]
k = np.load(r"D:\gravity\kalman_gravity\output_real\three_parts\kf_inputs.npz")
ssv_hr = k["ssv_hr"].astype(float)
dx = float(lon[1]-lon[0])*111320*np.cos(np.radians(lat.mean()))
dy = float(lat[1]-lat[0])*111320

c = np.where(bathy < 0, 0.0688, 0.1120)
boug = c*bathy
hp = boug - gaussian_filter(boug, 12.0)

diff = g15 - ssv_hr
m = np.isfinite(diff)
print("corr(diff, terrain_hp) =", np.corrcoef(diff[m], hp[m])[0,1])
print("std diff =", np.std(diff[m]), "std hp =", np.std(hp[m]))

for lo, hi in ((2e3,5e3),(5e3,10e3),(10e3,20e3)):
    a = bandpass(diff, dx, dy, lo, hi)
    b = bandpass(hp, dx, dy, lo, hi)
    print(f"{lo/1e3:.0f}-{hi/1e3:.0f} km: corr(diff,hp)={np.corrcoef(a[m],b[m])[0,1]:.3f}, "
          f"std diff={np.std(a[m]):.2f}, std hp={np.std(b[m]):.2f}")
