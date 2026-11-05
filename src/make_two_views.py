"""第1页：全球重力图 + 剖面波长 + 1′ 分辨率。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, RED, GOLD, BLUE, GRAY = "#0B2545", "#C0392B", "#C9A227", "#7FA8C9", "#90A4AE"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})

ds = xr.open_dataset(ROOT / "data/ssv32/grav_33.1.nc")
gv = ds["z"].values[::20, ::20]
glon = ds["lon"].values[::20]; glat = ds["lat"].values[::20]

d = np.load(OUT / "validation_G_fields.npz")
lon, lat = d["lon"], d["lat"]
j = int(np.argmin(np.abs(lat - 35.5)))
sel = (lon >= 136) & (lon <= 148)
x = lon[sel]
p15 = d["ckf"][j, sel]
dx = float(np.median(np.diff(x))) * 111.32 * np.cos(np.radians(35.5))
p1 = gaussian_filter1d(p15, 20.0 / dx)


def bp(sig, lo, hi):
    E = np.fft.rfft(sig)
    f = np.fft.rfftfreq(len(sig), d=dx)
    k = 2 * np.pi * f
    return np.fft.irfft(E * (1/(1+(k*lo)**4)) * (1-1/(1+(k*hi)**4)), n=len(sig))


long = bp(p15, 1e9, 100); mid = bp(p15, 20, 100); short = bp(p15, 5, 20)

fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.0), dpi=220,
                         gridspec_kw={"width_ratios": [1.25, 1.2, 1.0],
                                      "wspace": 0.24})
ax = axes[0]
im = ax.imshow(gv, cmap="RdBu_r", origin="lower",
               extent=[glon[0], glon[-1], glat[0], glat[-1]],
               vmin=-100, vmax=100, aspect="auto")
ax.set_title("(a) 全球海洋重力异常", fontsize=9, pad=3)
ax.set_xlabel("经度 (°E)", fontsize=7); ax.set_ylabel("纬度 (°N)", fontsize=7)
ax.tick_params(labelsize=6)
fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02).set_label("", fontsize=6)

ax = axes[1]
ax.plot(x, p15, color=NAVY, lw=1.3, label="Δg")
ax.plot(x, long, color=GRAY, lw=1.0, label="长波")
ax.plot(x, mid, color=BLUE, lw=0.9, label="中波")
ax.plot(x, short, color=RED, lw=0.8, label="短波")
ax.set_xlabel("经度 (°E)", fontsize=7); ax.set_ylabel("")
ax.set_title("(b) 同一条剖面：波长组成", fontsize=9)
ax.legend(fontsize=6, frameon=False, ncol=2); ax.grid(alpha=0.2)
ax.tick_params(labelsize=6)
ax.text(0.02, 0.93, r"$g=g_L+g_M+g_S$", transform=ax.transAxes,
        fontsize=8, color=NAVY, bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
ax.text(0.57, 0.72, r"$g(x)=\sum_k[a_k\cos(kx)+b_k\sin(kx)]$" + "\n" + "k 小 → 长波；k 大 → 短波",
        transform=ax.transAxes, fontsize=6.2, color=NAVY, ha="left", va="top",
        bbox=dict(fc="white", ec="#C9D3DF", alpha=0.92, pad=2))

ax = axes[2]
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.plot([0.2, 0.8], [0.48, 0.48], color=NAVY, lw=1.2)
ax.plot([0.2, 0.8], [0.48, 0.48], "o", color=RED, ms=6)
ax.annotate("", xy=(0.8, 0.62), xytext=(0.2, 0.62),
            arrowprops=dict(arrowstyle="<->", color=RED, lw=1.1))
ax.text(0.5, 0.70, "1′ ≈ 1.85 km", ha="center", fontsize=11, color=RED,
        fontweight="bold")
ax.text(0.5, 0.86, "1′ 数据点间距", ha="center", fontsize=9, color=NAVY,
        fontweight="bold")
ax.text(0.5, 0.34, "相邻两个数据点的距离", ha="center", fontsize=8,
        color="#44546A")
ax.text(0.5, 0.22, "纬向距离 = 1.85·cosφ km", ha="center", fontsize=8,
        color="#44546A")
ax.set_title("(c) 1′ 数据点间距", fontsize=9)

fig.tight_layout()
fig.savefig(OUT / "fig_new_two_views.png", bbox_inches="tight")
print("saved")
