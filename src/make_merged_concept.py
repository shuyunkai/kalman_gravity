"""合并页：场与波长 + 1′/15″ 分辨率 + 5–20 km 缺口。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, RED, GOLD, BLUE, GRAY = "#0B2545", "#C0392B", "#C9A227", "#7FA8C9", "#90A4AE"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})

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
    lp = 1 / (1 + (k * lo) ** 4)
    hp = 1 - 1 / (1 + (k * hi) ** 4)
    return np.fft.irfft(E * lp * hp, n=len(sig))


long = bp(p15, 1e9, 100)
mid = bp(p15, 20, 100)
short = bp(p15, 5, 20)

fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.0), dpi=220,
                         gridspec_kw={"width_ratios": [1.35, 1.05, 1.0],
                                      "wspace": 0.14})
ax = axes[0]
ax.plot(x, p15, color=NAVY, lw=1.4, label="Δg")
ax.plot(x, long, color=GRAY, lw=1.0, label="长波")
ax.plot(x, mid, color=BLUE, lw=0.9, label="中波")
ax.plot(x, short, color=RED, lw=0.8, label="短波")
ax.set_xlabel("经度 (°E)", fontsize=7); ax.set_ylabel("mGal", fontsize=7)
ax.set_title("(a) 视角一：波长组成", fontsize=9)
ax.legend(fontsize=6, frameon=False, ncol=2); ax.grid(alpha=0.2)
ax.tick_params(labelsize=6)
ax.text(0.02, 0.93, r"$g=g_L+g_M+g_S$", transform=ax.transAxes,
        fontsize=8, color=NAVY, bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))

ax = axes[1]
m = (x >= 142) & (x <= 146)
ax.plot(x[m], p15[m], color=NAVY, lw=1.4, label="15″ 场")
ax.plot(x[m], p1[m], color=GRAY, lw=1.4, ls="--", label="1′ 产品")
ax.fill_between(x[m], p1[m], p15[m], color=RED, alpha=0.18)
ax.set_xlabel("经度 (°E)", fontsize=7); ax.set_ylabel("mGal", fontsize=7)
ax.set_title("(b) 视角二：1′ 产品与 15″ 分辨率", fontsize=9)
ax.legend(fontsize=6, frameon=False); ax.grid(alpha=0.2); ax.tick_params(labelsize=6)

ax = axes[2]
ax.set_xscale("log"); ax.set_xlim(1, 100); ax.set_ylim(0, 1)
ax.axvspan(5, 20, color=RED, alpha=0.12)
ax.axvline(0.93, color=BLUE, lw=1.1, ls="--")
ax.axvline(3.7, color=NAVY, lw=1.1, ls="--")
ax.axvline(12, color=GRAY, lw=0.9, ls=":")
ax.axvline(20, color=GRAY, lw=0.9, ls=":")
ax.text(1.1, 0.62, "15″\n0.93", fontsize=6.5, color=BLUE)
ax.text(3.9, 0.62, "1′\n3.7", fontsize=6.5, color=NAVY)
ax.text(15, 0.10, "5–20 km", ha="center", fontsize=7, color=RED)
ax.text(12, 0.82, "1′ 产品有效 12–20", fontsize=6.5, color=GRAY)
ax.annotate("5–12 被平滑", xy=(8, 0.36), xytext=(1.2, 0.36),
            fontsize=6.5, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
ax.set_xlabel("波长 (km)", fontsize=7)
ax.set_title("(c) 分辨率与 5–20 km", fontsize=9)
ax.set_yticks([]); ax.grid(alpha=0.15); ax.tick_params(labelsize=6)

fig.tight_layout()
fig.savefig(OUT / "fig_new_field_resolution_merged.png", bbox_inches="tight")
print("saved")
