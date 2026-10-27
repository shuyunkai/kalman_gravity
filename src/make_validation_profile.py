"""验证页：留出航次剖面 + 误差剖面 + 密度散点 + 误差直方图。"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"output_real/three_parts/newdeck"
NAVY, RED, GRAY = "#0B2545", "#C0392B", "#90A4AE"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})

d = np.load(f"{ROOT}/validation_G_fields.npz")
lon, lat = d["lon"], d["lat"]
val = np.asarray(d["val"], float)
ix = np.clip(np.searchsorted(lon, val[:, 0]), 0, len(lon)-1)
iy = np.clip(np.searchsorted(lat, val[:, 1]), 0, len(lat)-1)
ssv = d["ssv"][iy, ix]
ckf = d["ckf"][iy, ix]

n = 600
v = val[:n]
dist = np.r_[0, np.cumsum(np.hypot(np.diff(v[:, 0]) * 111 * np.cos(np.radians(v[:, 1].mean())),
                                   np.diff(v[:, 1]) * 111))]
fig, axes = plt.subplots(2, 2, figsize=(12.8, 5.4), dpi=220)

ax = axes[0, 0]
ax.plot(dist, v[:, 2], color="k", lw=1.2, label="船测")
ax.plot(dist, ssv[:n], color=GRAY, lw=1.2, ls="--", label="双线性 15″")
ax.plot(dist, ckf[:n], color=NAVY, lw=1.4, label="相干保持 KF")
ax.set_xlabel("沿测线距离 (km)", fontsize=8)
ax.set_ylabel("重力异常 (mGal)", fontsize=8)
ax.set_title("(a) 留出航次剖面", fontsize=10)
ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.2)
ax.tick_params(labelsize=7)

ax = axes[0, 1]
ax.plot(dist, ssv[:n] - v[:, 2], color=GRAY, lw=1.1, ls="--", label="双线性 − 船测")
ax.plot(dist, ckf[:n] - v[:, 2], color=NAVY, lw=1.2, label="相干保持 KF − 船测")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("沿测线距离 (km)", fontsize=8)
ax.set_ylabel("误差 (mGal)", fontsize=8)
ax.set_title("(b) 剖面误差", fontsize=10)
ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.2)
ax.tick_params(labelsize=7)

ax = axes[1, 0]
hb = ax.hexbin(val[:, 2], ckf, gridsize=60, cmap="magma", mincnt=1, bins="log")
lo, hi = np.percentile(val[:, 2], [1, 99])
ax.plot([lo, hi], [lo, hi], "--", color="cyan", lw=1)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("船测 (mGal)", fontsize=8); ax.set_ylabel("相干保持 KF (mGal)", fontsize=8)
ax.set_title("(c) 船测—模型密度散点", fontsize=10)
ax.tick_params(labelsize=7)
cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
cb.set_label("log10(点数)", fontsize=7)

ax = axes[1, 1]
bins = np.linspace(-20, 20, 41)
e1 = ssv - val[:, 2]; e2 = ckf - val[:, 2]
r1 = float(np.sqrt(np.mean(e1 ** 2))); r2 = float(np.sqrt(np.mean(e2 ** 2)))
ax.hist(e1, bins=bins, color=GRAY, alpha=0.65, label=f"双线性  RMSE {r1:.2f}")
ax.hist(e2, bins=bins, color=NAVY, alpha=0.7, label=f"相干保持 KF  RMSE {r2:.2f}")
ax.set_xlabel("误差 (mGal)", fontsize=8); ax.set_ylabel("点数", fontsize=8)
ax.set_title("(d) 留出航次误差分布", fontsize=10)
ax.legend(fontsize=7, frameon=False); ax.grid(axis="y", alpha=0.2)
ax.tick_params(labelsize=7)

fig.tight_layout()
fig.savefig(f"{ROOT}/fig_new_validation_profile.png", bbox_inches="tight")
print("saved")
