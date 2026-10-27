"""新增页：三种常用 15″ 场与相干保持 KF 的场/差值对比（日本区）。"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"output_real/three_parts/newdeck"
NAVY, RED, GOLD = "#0B2545", "#C0392B", "#C9A227"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 8})

d = np.load(f"{ROOT}/validation_G_fields.npz")
lon, lat = d["lon"], d["lat"]
ssv, freq, lsc, ckf = d["ssv"], d["freq"], d["lsc"], d["ckf"]
i0 = np.searchsorted(lat, 34.0); i1 = np.searchsorted(lat, 37.0)
j0 = np.searchsorted(lon, 142.0); j1 = np.searchsorted(lon, 145.0)
ext = [lon[j0], lon[j1], lat[i0], lat[i1]]

fields = [("(a) 双线性", ssv), ("(b) 频域融合", freq),
          ("(c) LSC", lsc), ("(d) 相干保持 KF", ckf)]
diffs = [("(e) CKF − 双线性", ckf - ssv), ("(f) CKF − 频域", ckf - freq),
         ("(g) CKF − LSC", ckf - lsc)]

fig, axes = plt.subplots(2, 4, figsize=(13.4, 5.4), dpi=220)
for ax, (name, fld) in zip(axes[0], fields):
    z = fld[i0:i1, j0:j1]
    v = np.nanpercentile(np.abs(z), 98)
    im = ax.imshow(z, cmap="RdBu_r", origin="lower", vmin=-v, vmax=v,
                   extent=ext, aspect="auto")
    ax.set_title(name, fontsize=9)
    ax.set_xlabel("经度 (°E)", fontsize=7); ax.set_ylabel("纬度 (°N)", fontsize=7)
    ax.tick_params(labelsize=6)
im_top = im

for ax, (name, fld) in zip(axes[1, :3], diffs):
    z = fld[i0:i1, j0:j1]
    v = np.nanpercentile(np.abs(z), 98)
    im = ax.imshow(z, cmap="PuOr_r", origin="lower", vmin=-v, vmax=v,
                   extent=ext, aspect="auto")
    ax.set_title(name, fontsize=9)
    ax.set_xlabel("经度 (°E)", fontsize=7); ax.set_ylabel("纬度 (°N)", fontsize=7)
    ax.tick_params(labelsize=6)
im_bot = im

ax = axes[1, 3]
vals = [np.sqrt(np.nanmean((ckf - ssv) ** 2)),
        np.sqrt(np.nanmean((ckf - freq) ** 2)),
        np.sqrt(np.nanmean((ckf - lsc) ** 2))]
names = ["双线性", "频域", "LSC"]
ax.bar(names, vals, color=[NAVY, "#7FA8C9", GOLD], width=0.55)
for i, v in enumerate(vals):
    ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=7)
ax.set_ylabel("差值 RMS (mGal)", fontsize=8)
ax.set_title("(h) 差值 RMS", fontsize=9)
ax.tick_params(labelsize=7); ax.grid(axis="y", alpha=0.2)

fig.subplots_adjust(left=0.05, right=0.90, top=0.90, bottom=0.11,
                    wspace=0.28, hspace=0.38)
cax1 = fig.add_axes([0.915, 0.55, 0.008, 0.30])
fig.colorbar(im_top, cax=cax1).set_label("mGal", fontsize=7)
cax2 = fig.add_axes([0.05, 0.015, 0.18, 0.010])
fig.colorbar(im_bot, cax=cax2, orientation="horizontal").set_label("差值 mGal", fontsize=6)
fig.savefig(f"{ROOT}/fig_new_methods_diff.png", bbox_inches="tight")
print("saved")
