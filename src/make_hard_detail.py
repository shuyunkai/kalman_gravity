"""困难页细节图：块平均、频带、船测覆盖、多尺度剖面。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).parent))
from scale_decomposition import bandpass
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 8})

d = np.load(OUT / "validation_G_fields.npz")
lon, lat = d["lon"], d["lat"]
ssv, ckf = d["ssv"], d["ckf"]
val = np.asarray(d["val"], float)
dx = (lon[1]-lon[0])*111320*np.cos(np.radians(lat.mean()))
dy = (lat[1]-lat[0])*111320

fig, axes = plt.subplots(2, 2, figsize=(13.4, 5.2), dpi=220)
# (a) block grid
ax = axes[0, 0]
i0, i1, j0, j1 = 2000, 2200, 1600, 1800
z = ckf[i0:i1, j0:j1]
vv = np.nanpercentile(np.abs(z), 98)
ax.imshow(z, cmap="RdBu_r", origin="lower", vmin=-vv, vmax=vv, aspect="auto")
for k in range(0, 200, 4):
    ax.axhline(k, color="k", lw=0.25, alpha=0.5)
    ax.axvline(k, color="k", lw=0.25, alpha=0.5)
ax.set_title("(a) 1′ 块 = 4×4 个 15″ 格点", fontsize=8)
ax.set_xticks([]); ax.set_yticks([])

# (b) 5-20 band energy
ax = axes[0, 1]
e1 = np.std(bandpass(gaussian_filter(ssv, 8e3/dx), dx, dy, 5e3, 20e3)[i0:i1, j0:j1])
e2 = np.std(bandpass(ckf, dx, dy, 5e3, 20e3)[i0:i1, j0:j1])
ax.bar([0, 1], [e1, e2], color=["#90A4AE", "#C0392B"], width=0.5)
ax.set_xticks([0, 1]); ax.set_xticklabels(["1′ 产品", "15″ 场"], fontsize=7)
ax.set_ylabel("5–20 km 标准差 (mGal)", fontsize=7)
ax.set_title("(b) 5–20 km 频带能量", fontsize=8)
ax.grid(axis="y", alpha=0.2); ax.tick_params(labelsize=6)

# (c) ship residual histogram
ax = axes[1, 0]
ixv = np.clip(np.searchsorted(lon, val[:, 0]), 0, len(lon)-1)
iyv = np.clip(np.searchsorted(lat, val[:, 1]), 0, len(lat)-1)
e = ckf[iyv, ixv] - val[:, 2]
ax.hist(e, bins=40, color="#C0392B", alpha=0.75)
ax.axvline(0, color="k", lw=0.8)
ax.set_title("(c) 船测残差进入反演", fontsize=8)
ax.set_xlabel("误差 (mGal)", fontsize=7)
ax.tick_params(labelsize=6)

# (d) profile decomposition
ax = axes[1, 1]
x = lon[1400:2200]
p = ckf[2100, 1400:2200]
long = bandpass(p, dx, dy, 1e9, 100e3) if False else None
# 1D band decomposition
E = np.fft.rfft(p); ff = np.fft.rfftfreq(len(p), d=dx); kk = 2*np.pi*ff
def bp(lo, hi):
    lp = 1/(1+(kk*lo)**4); hp = 1-1/(1+(kk*hi)**4)
    return np.fft.irfft(E*lp*hp, n=len(p))
l = bp(1e9, 100e3); mid = bp(20e3, 100e3); sh = bp(5e3, 20e3)
ax.plot(x, p, color="k", lw=1, label="剖面")
ax.plot(x, l, color="#90A4AE", lw=1, label=">100 km")
ax.plot(x, mid, color="#1F77B4", lw=0.9, label="20–100")
ax.plot(x, sh, color="#C0392B", lw=0.9, label="5–20")
ax.set_title("(d) 多尺度成分叠加", fontsize=8)
ax.legend(fontsize=5.5, frameon=False, ncol=2)
ax.set_xticks([]); ax.set_yticks([])

fig.tight_layout()
fig.savefig(OUT / "fig_new_hard_detail.png", bbox_inches="tight")
print("saved")
