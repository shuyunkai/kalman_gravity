"""前两页：同一条剖面看波长；同一条剖面看 1′ 产品与 15″ 分辨率。"""
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

# page1: one profile -> long/mid/short + formulas
fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0), dpi=220,
                         gridspec_kw={"width_ratios": [1.5, 0.9], "wspace": 0.10})
ax = axes[0]
ax.plot(x, p15, color=NAVY, lw=1.5, label="重力剖面 Δg")
ax.plot(x, long, color=GRAY, lw=1.1, label="长波")
ax.plot(x, mid, color=BLUE, lw=1.0, label="中波")
ax.plot(x, short, color=RED, lw=0.9, label="短波")
ax.set_xlabel("经度 (°E)", fontsize=8); ax.set_ylabel("mGal", fontsize=8)
ax.set_title("(a) 同一条重力剖面：不同波长成分", fontsize=10)
ax.legend(fontsize=7, frameon=False, ncol=2); ax.grid(alpha=0.2)
ax.tick_params(labelsize=7)

ax = axes[1]; ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.02, 0.94, "傅里叶展开", fontsize=10, color=GOLD, fontweight="bold")
ax.text(0.02, 0.84, r"$g=g_L+g_M+g_S$", fontsize=13, color=NAVY)
ax.text(0.02, 0.72, r"长波 $g_L=A_L\sin(2\pi x/\lambda_L+\phi_L)$",
        fontsize=7.5, color=GRAY)
ax.text(0.02, 0.60, r"中波 $g_M=A_M\sin(2\pi x/\lambda_M+\phi_M)$",
        fontsize=7.5, color=BLUE)
ax.text(0.02, 0.48, r"短波 $g_S=A_S\sin(2\pi x/\lambda_S+\phi_S)$",
        fontsize=7.5, color=RED)
ax.text(0.02, 0.35, "λ：相邻波峰距离", fontsize=8, color=RED)
ax.text(0.02, 0.24, "λ 越大越平缓；\nλ 越小越尖锐。", fontsize=8, color="#44546A")
fig.tight_layout()
fig.savefig(OUT / "fig_new_profile_wave.png", bbox_inches="tight")

# page2: same profile, 1' product vs 15" field, zoom
fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0), dpi=220,
                         gridspec_kw={"width_ratios": [1.25, 1.15], "wspace": 0.12})
ax = axes[0]
ax.plot(x, p15, color=NAVY, lw=1.4, label="15″ 场")
ax.plot(x, p1, color=GRAY, lw=1.5, ls="--", label="1′ 产品（平滑）")
ax.set_xlabel("经度 (°E)", fontsize=8); ax.set_ylabel("mGal", fontsize=8)
ax.set_title("(a) 同一条剖面：1′ 产品与 15″ 场", fontsize=10)
ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.2); ax.tick_params(labelsize=7)

ax = axes[1]
m = (x >= 142) & (x <= 146)
ax.plot(x[m], p15[m], color=NAVY, lw=1.5, label="15″ 场")
ax.plot(x[m], p1[m], color=GRAY, lw=1.5, ls="--", label="1′ 产品（平滑）")
ax.fill_between(x[m], p1[m], p15[m], color=RED, alpha=0.18)
ax.set_xlabel("经度 (°E)", fontsize=8); ax.set_ylabel("mGal", fontsize=8)
ax.set_title("(b) 局部：1′ 产品抹掉的 5–20 km 短波", fontsize=10)
ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.2); ax.tick_params(labelsize=7)
ax.text(0.03, 0.05, "1′ 网格 λ_min≈3.7 km，但产品有效 12–20 km", transform=ax.transAxes,
        fontsize=7, color=RED)
fig.tight_layout()
fig.savefig(OUT / "fig_new_profile_resolution.png", bbox_inches="tight")
print("saved")
