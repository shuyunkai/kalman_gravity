"""CKF 对比页：标准 KF 单点更新 vs CKF 邻域修正（5–20 km）。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).parent))
from scale_decomposition import bandpass

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, RED, BLUE = "#0B2545", "#C0392B", "#7FA8C9"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})

d = np.load(OUT / "validation_G_fields.npz")
lon, lat = d["lon"], d["lat"]
kf, ckf = d["kf"], d["ckf"]
dx = (lon[1]-lon[0])*111320*np.cos(np.radians(lat.mean()))
dy = (lat[1]-lat[0])*111320
g1 = bandpass(kf, dx, dy, 5e3, 20e3)
g2 = bandpass(ckf, dx, dy, 5e3, 20e3)
i, j = 2100, 1800
sub = (slice(i-80, i+80), slice(j-80, j+80))
z1, z2 = g1[sub], g2[sub]
row = int(np.argmax(np.std(z2-z1, axis=1)))
v = np.nanpercentile(np.abs(np.r_[z1.ravel(), z2.ravel()]), 98)

fig = plt.figure(figsize=(13.4, 5.0), dpi=220)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.12)

ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(z1, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
ax.axhline(row, color="k", lw=1.0, ls="--")
ax.text(4, row+5, "剖面位置", fontsize=6, color="k",
        bbox=dict(fc="white", ec="none", alpha=0.75, pad=1))
ax.set_title("(a) 标准 KF：5–20 km", fontsize=9)
ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)

ax = fig.add_subplot(gs[0, 1])
im = ax.imshow(z2, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
ax.axhline(row, color="k", lw=1.0, ls="--")
ax.text(4, row+5, "剖面位置", fontsize=6, color="k",
        bbox=dict(fc="white", ec="none", alpha=0.75, pad=1))
ax.set_title("(b) 相干保持 KF：5–20 km", fontsize=9)
ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)

ax = fig.add_subplot(gs[0, 2])
xkm = np.arange(z1.shape[1]) * dx / 1000.0
ax.plot(xkm, z1[row, :], color=NAVY, lw=1.4, label="标准 KF")
ax.plot(xkm, z2[row, :], color=RED, lw=1.4, label="相干保持 KF")
ax.fill_between(xkm, z1[row, :], z2[row, :], color=RED, alpha=0.14, label="两者差值")
ax.set_title("(c) 5–20 km 短波剖面", fontsize=9)
ax.set_xlabel("沿剖面距离 (km)", fontsize=7); ax.set_ylabel("mGal", fontsize=7)
ax.legend(fontsize=6.2, frameon=False, loc="upper right"); ax.grid(alpha=0.2)
ax.tick_params(labelsize=6)
ax.set_ylim(-18, 18); ax.set_yticks([-15,-10,-5,0,5,10,15])

fig.savefig(OUT / "fig_new_ckf_compare.png", bbox_inches="tight")
print("saved")
