"""验证页：船测点误差图 + 四种方法误差分布（替代场差值图）。"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"output_real/three_parts/newdeck"
NAVY, RED, GOLD, GRAY, BLUE = "#0B2545", "#C0392B", "#C9A227", "#90A4AE", "#7FA8C9"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})

d = np.load(f"{ROOT}/validation_G_fields.npz")
lon, lat = d["lon"], d["lat"]
val = np.asarray(d["val"], float)
ix = np.clip(np.searchsorted(lon, val[:, 0]), 0, len(lon)-1)
iy = np.clip(np.searchsorted(lat, val[:, 1]), 0, len(lat)-1)
methods = [("双线性", "ssv", GRAY), ("频域融合", "freq", BLUE),
           ("LSC", "lsc", GOLD), ("相干保持 KF", "ckf", NAVY)]
err = {name: d[key][iy, ix] - val[:, 2] for name, key, _ in methods}

fig = plt.figure(figsize=(13.4, 5.2), dpi=220)
gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.18)

# (a) ship-point error map
ax = fig.add_subplot(gs[0, 0])
k = 10
ext = [lon[k], lon[-k], lat[k], lat[-k]]
v = np.percentile(np.abs(d["ckf"][k:-k, k:-k]), 99)
im = ax.imshow(d["ckf"][k:-k, k:-k], cmap="RdBu_r", origin="lower",
               vmin=-v, vmax=v, extent=ext, aspect="auto", alpha=0.65)
e = np.abs(err["相干保持 KF"])
vmax = np.percentile(e, 99)
sc = ax.scatter(val[:, 0], val[:, 1], c=e, s=3, cmap="plasma",
                vmin=0, vmax=vmax, linewidths=0, alpha=0.8)
ax.set_xlim(138.0, 147.0); ax.set_ylim(34.0, 42.0)
ax.set_xlabel("经度 (°E)", fontsize=8); ax.set_ylabel("纬度 (°N)", fontsize=8)
ax.set_title("(a) 留出船测点误差（颜色为 |相干保持 KF−船测|）", fontsize=10)
ax.tick_params(labelsize=7)
cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
cb.set_label("|误差| (mGal)", fontsize=7); cb.ax.tick_params(labelsize=6)

# (b) error distribution
ax = fig.add_subplot(gs[0, 1])
data = [err[name][np.isfinite(err[name])] for name, _, _ in methods]
bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                medianprops=dict(color="black", lw=1.2))
for patch, (_, _, col) in zip(bp["boxes"], methods):
    patch.set_facecolor(col); patch.set_alpha(0.75)
for i, (name, _, _) in enumerate(methods, 1):
    rmse = np.sqrt(np.mean(data[i-1]**2))
    ax.text(i, np.percentile(data[i-1], 98)+0.8, f"{rmse:.2f}",
            ha="center", fontsize=8, color=RED, fontweight="bold")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(1, 5)); ax.set_xticklabels([m[0] for m in methods], fontsize=7)
ax.set_ylabel("模型 − 船测 (mGal)", fontsize=8)
ax.set_title("(b) 误差分布（箱线，数字为 RMSE）", fontsize=10)
ax.grid(axis="y", alpha=0.25); ax.tick_params(labelsize=7)
ax.set_ylim(-12, 12)

fig.tight_layout()
fig.savefig(f"{ROOT}/fig_new_validation_track.png", bbox_inches="tight")
print("saved")
