"""第 5 页：三种方法的缺点 + 共同短波不足。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from scale_decomposition import bandpass

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, RED, GOLD, PALE = "#0B2545", "#C0392B", "#C9A227", "#D6E2F0"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})


def spectrum(f, dx, dy, nb=60):
    F = np.fft.fft2(f); P = np.abs(F) ** 2
    kx = 2*np.pi*np.fft.fftfreq(f.shape[1], dx)
    ky = 2*np.pi*np.fft.fftfreq(f.shape[0], dy)
    KX, KY = np.meshgrid(kx, ky); K = np.hypot(KX, KY).ravel(); PP = P.ravel()
    lo, hi = 2*np.pi/500e3, 2*np.pi/5e3
    m = (K > lo) & (K < hi); K = K[m]; PP = PP[m]
    bins = np.logspace(np.log10(lo), np.log10(hi), nb)
    idx = np.digitize(K, bins); kk=[]; pp=[]
    for i in range(1, nb):
        q = idx == i
        if q.sum() > 5:
            kk.append(np.mean(K[q])); pp.append(np.mean(PP[q]))
    return 2*np.pi/np.array(kk)/1e3, np.array(pp)


d = np.load(OUT / "validation_G_fields.npz")
lon, lat, ssv = d["lon"], d["lat"], d["ssv"]
dx = (lon[1]-lon[0])*111320*np.cos(np.radians(lat.mean()))
dy = (lat[1]-lat[0])*111320
ext = [lon[0], lon[-1], lat[0], lat[-1]]
k = 4; sub = (slice(k, -k), slice(k, -k))

fig, axes = plt.subplots(2, 2, figsize=(13.4, 5.2), dpi=220)
ax = axes[0, 0]
i = np.argmin(np.abs(lat - 36.0))
xs1 = lon[::4]; ys1 = ssv[i, ::4]
ax.step(xs1, ys1, where="mid", color=NAVY, lw=1.3, marker="o", ms=2.5,
        label="1′ 格点")
ax.plot(lon, ssv[i], color=RED, lw=1.1, ls="--", label="15″ 插值")
x0, x1 = 141.6, 142.2
ax.set_xlim(x0, x1)
m = (lon >= x0) & (lon <= x1)
ax.set_ylim(np.nanmin(ssv[i, m]) - 5, np.nanmax(ssv[i, m]) + 5)
ax.text(0.03, 0.9, "两条线重合", transform=ax.transAxes, ha="left",
        fontsize=8, color=RED)
ax.set_title("(a) 插值：只加密，不增短波", fontsize=10)
ax.set_xlabel("经度 (°E)", fontsize=8); ax.set_ylabel("mGal", fontsize=8)
ax.legend(fontsize=7, frameon=False); ax.tick_params(labelsize=7)
ax.grid(alpha=0.2)

ax = axes[0, 1]
pad = np.pad(ssv, 64, mode="reflect")
bp_pad = bandpass(pad, dx, dy, 5e3, 20e3)[64:-64, 64:-64]
bp = bandpass(ssv, dx, dy, 5e3, 20e3)
edge = bp - bp_pad
ve = np.nanpercentile(np.abs(edge), 98)
im = ax.imshow(edge[sub], cmap="PuOr_r", origin="lower", vmin=-ve, vmax=ve,
               extent=ext, aspect="auto")
ax.set_title("(b) 频域融合：边界振铃（残差）", fontsize=10)
ax.set_xlabel("经度 (°E)", fontsize=8); ax.set_ylabel("纬度 (°N)", fontsize=8)
ax.tick_params(labelsize=7)
ax.text(0.5, 0.52, "内部≈0", transform=ax.transAxes, ha="center",
        fontsize=9, color="#5B6573")
ax.text(0.02, 0.94, "边界残差", transform=ax.transAxes, fontsize=8,
        color=RED, fontweight="bold")
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("边界残差 (mGal)", fontsize=7)

ax = axes[1, 0]
r = np.linspace(0, 100, 200); a = 30
ax.plot(r, np.exp(-r**2/(2*a**2)), color=NAVY, lw=1.5, label="高斯")
ax.plot(r, np.exp(-r/a), color=RED, lw=1.5, ls="--", label="指数")
ax.plot(r, (1+r/a)*np.exp(-r/a), color=GOLD, lw=1.5, ls=":", label="Matérn")
ax.set_xlabel("距离 (km)", fontsize=8); ax.set_ylabel("协方差", fontsize=8)
ax.set_title("(c) LSC：协方差函数选择敏感", fontsize=10)
ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.2); ax.tick_params(labelsize=7)

ax = axes[1, 1]
lam, p = spectrum(ssv, dx, dy)
ax.loglog(lam, p/p.max(), color=NAVY, lw=1.4)
ax.axvspan(5, 20, color="#F2B8B5", alpha=0.25)
ax.set_xlabel("波长 (km)", fontsize=8); ax.set_ylabel("归一化功率（/最大值）", fontsize=8)
ax.set_title("(d) 三者共同：5–20 km 能量弱", fontsize=10)
ax.grid(alpha=0.2, which="both"); ax.tick_params(labelsize=7)
ax.text(8, 0.25, "5–20 km", color=RED, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig_new_cons_3methods.png", bbox_inches="tight")
print("saved")
