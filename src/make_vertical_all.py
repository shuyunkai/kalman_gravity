"""纵向精度页：逐轮/RTS、分波长、分水深、分地形梯度。"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from three_method_compare import load_bathy_15s

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, RED, GOLD, GRAY, BLUE = "#0B2545", "#C0392B", "#C9A227", "#90A4AE", "#7FA8C9"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 8})

d = np.load(OUT / "validation_G_fields.npz")
lon, lat = d["lon"], d["lat"]
val = np.asarray(d["val"], float)
ix = np.clip(np.searchsorted(lon, val[:, 0]), 0, len(lon)-1)
iy = np.clip(np.searchsorted(lat, val[:, 1]), 0, len(lat)-1)
methods = [("双线性", "ssv", GRAY), ("频域融合", "freq", BLUE),
           ("LSC", "lsc", GOLD), ("相干保持 KF", "ckf", NAVY)]
err = {name: d[key][iy, ix] - val[:, 2] for name, key, _ in methods}


def heatmap(ax, mat, rows, cols, title, vmax):
    im = ax.imshow(mat, cmap="RdYlGn_r", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=7)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=7)
    ax.set_title(title, fontsize=9)
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5,
                    color="white" if mat[i, j] > vmax*0.65 else "#1A1A1A")
    return im

fig, axes = plt.subplots(2, 2, figsize=(13.4, 5.6), dpi=220)

# (a) iteration / RTS
vals = np.load(OUT / "vertical_rmse_G.npz")["vals"]
labels = ["先验"] + [str(i) for i in range(1, 7)] + ["RTS"]
ax = axes[0, 0]
ax.plot(range(len(vals)), vals, "o-", color=NAVY, lw=1.5, ms=4)
ax.axhline(vals[1], color=GRAY, ls="--", lw=0.8)
for i, v in enumerate(vals):
    if i in (0, 1, len(vals)-1):
        ax.text(i, v+0.08, f"{v:.2f}", ha="center", fontsize=6.5)
ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("船测点 RMSE (mGal)", fontsize=8)
ax.set_title("(a) 逐轮/RTS：先验→6轮→RTS", fontsize=9)
ax.grid(alpha=0.2)

# (b) along-track band error
n = 600
v = val[:n]
dist = np.r_[0, np.cumsum(np.hypot(
    np.diff(v[:, 0])*111*np.cos(np.radians(v[:, 1].mean())),
    np.diff(v[:, 1])*111))]*1000
bands = [(5e3, 20e3), (20e3, 50e3), (50e3, 100e3)]
band_names = ["5–20", "20–50", "50–100"]
band_err = {}
for name, key, _ in methods:
    e = d[key][iy, ix][:n] - v[:, 2]
    vals_b = []
    for lo, hi in bands:
        E = np.fft.rfft(e); f = np.fft.rfftfreq(len(e), d=np.median(np.diff(dist)))
        k = 2*np.pi*f
        lp = 1/(1 + (k*lo)**4); hp = 1 - 1/(1 + (k*hi)**4)
        vals_b.append(float(np.sqrt(np.mean(np.fft.irfft(E*lp*hp, n=len(e))**2))))
    band_err[name] = vals_b
ax = axes[0, 1]
M = np.array([band_err[name] for name, _, _ in methods])
im = heatmap(ax, M, [m[0] for m in methods],
             [f"{b} km" for b in band_names],
             "(b) 分波长：沿剖面 RMSE", np.nanmax(M))
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("mGal", fontsize=7)

# (c) depth bins
bathy = load_bathy_15s((132.0, 152.0), (26.0, 42.0), lon, lat)
depth = bathy[iy, ix]
dq = np.nanpercentile(depth, [33, 66])
depth_bins = [(-np.inf, dq[0]), (dq[0], dq[1]), (dq[1], np.inf)]
depth_names = ["浅", "中", "深"]
ax = axes[1, 0]
vals_d = []
for name, key, col in methods:
    row = []
    for lo, hi in depth_bins:
        m = (depth >= lo) & (depth < hi)
        row.append(float(np.sqrt(np.mean(err[name][m]**2))) if m.any() else np.nan)
    vals_d.append(row)
M = np.array(vals_d)
im = heatmap(ax, M, [m[0] for m in methods], depth_names,
             "(c) 分水深：船测点 RMSE", np.nanmax(M))
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("mGal", fontsize=7)

# (d) terrain gradient bins
gy, gx = np.gradient(bathy)
grad = np.hypot(gx, gy)[iy, ix]
qs = np.nanpercentile(grad, [33, 66])
grad_bins = [(-np.inf, qs[0]), (qs[0], qs[1]), (qs[1], np.inf)]
grad_names = ["平缓", "中等", "陡变"]
ax = axes[1, 1]
vals_g = []
for name, key, col in methods:
    row = []
    for lo, hi in grad_bins:
        m = (grad >= lo) & (grad < hi)
        row.append(float(np.sqrt(np.mean(err[name][m]**2))) if m.any() else np.nan)
    vals_g.append(row)
M = np.array(vals_g)
im = heatmap(ax, M, [m[0] for m in methods], grad_names,
             "(d) 分地形梯度：船测点 RMSE", np.nanmax(M))
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("mGal", fontsize=7)

fig.tight_layout()
fig.savefig(OUT / "fig_new_vertical_accuracy.png", bbox_inches="tight")
print("saved")
