"""验证页：留出航次剖面（四种方法）+ 误差剖面 + 密度散点。"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"output_real/three_parts/newdeck"
NAVY, RED, GRAY, BLUE, GOLD = "#0B2545", "#C0392B", "#90A4AE", "#7FA8C9", "#C9A227"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})

d = np.load(f"{ROOT}/validation_G_fields.npz")
lon, lat = d["lon"], d["lat"]
val = np.asarray(d["val"], float)
ix = np.clip(np.searchsorted(lon, val[:, 0]), 0, len(lon)-1)
iy = np.clip(np.searchsorted(lat, val[:, 1]), 0, len(lat)-1)
fields = {
    "双线性": d["ssv"][iy, ix],
    "频域融合": d["freq"][iy, ix],
    "LSC": d["lsc"][iy, ix],
    "相干保持 KF": d["ckf"][iy, ix],
}
styles = {
    "双线性": (GRAY, "--"),
    "频域融合": (BLUE, "-."),
    "LSC": (GOLD, ":"),
    "相干保持 KF": (NAVY, "-"),
}

n = 600
v = val[:n]
dist = np.r_[0, np.cumsum(np.hypot(
    np.diff(v[:, 0]) * 111 * np.cos(np.radians(v[:, 1].mean())),
    np.diff(v[:, 1]) * 111))]

fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.4), dpi=220)

ax = axes[0]
ax.plot(dist, v[:, 2], color="k", lw=1.2, label="船测")
for name, fld in fields.items():
    col, ls = styles[name]
    ax.plot(dist, fld[:n], color=col, lw=1.1, ls=ls, label=name)
ax.set_xlabel("沿测线距离 (km)", fontsize=8)
ax.set_ylabel("重力异常 (mGal)", fontsize=8)
ax.set_title("(a) 留出航次剖面", fontsize=10)
ax.legend(fontsize=6.2, frameon=False, ncol=2); ax.grid(alpha=0.2)
ax.tick_params(labelsize=7)

ax = axes[1]
for name, fld in fields.items():
    col, ls = styles[name]
    ax.plot(dist, fld[:n] - v[:, 2], color=col, lw=1.0, ls=ls, label=f"{name} − 船测")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("沿测线距离 (km)", fontsize=8)
ax.set_ylabel("误差 (mGal)", fontsize=8)
ax.set_title("(b) 剖面误差", fontsize=10)
ax.legend(fontsize=6.2, frameon=False); ax.grid(alpha=0.2)
ax.tick_params(labelsize=7)

ax = axes[2]
hb = ax.hexbin(val[:, 2], fields["相干保持 KF"], gridsize=60,
               cmap="magma", mincnt=1, bins="log")
lo, hi = np.percentile(val[:, 2], [1, 99])
ax.plot([lo, hi], [lo, hi], "--", color="cyan", lw=1)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("船测 (mGal)", fontsize=8)
ax.set_ylabel("相干保持 KF (mGal)", fontsize=8)
ax.set_title("(c) 船测—模型密度散点", fontsize=10)
ax.tick_params(labelsize=7)
cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
cb.set_label("log10(点数)", fontsize=7)

fig.tight_layout()
fig.savefig(f"{ROOT}/fig_new_validation_profile.png", bbox_inches="tight")
print("saved")
