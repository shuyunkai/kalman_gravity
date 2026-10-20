"""KF 原理示意图：多尺度状态分解 + 谱域反演增益。"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.ndimage import gaussian_filter

FONT = Path(r"D:\gravity\fonts\SourceHanSansSC-Regular.otf")
if FONT.exists():
    fm.fontManager.addfont(str(FONT))
plt.rcParams["font.sans-serif"] = ["Source Han Sans SC", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(r"D:\gravity\kalman_gravity\output_real\three_parts")

# ---------------- 图 1：多尺度状态分解
Y, X = np.mgrid[0:300, 0:300]
c = np.sin(2 * np.pi * X / 120) + np.cos(2 * np.pi * Y / 140)
m = 0.6 * np.sin(2 * np.pi * X / 40 + 1.0) + 0.4 * np.cos(2 * np.pi * Y / 34)
f = 0.3 * np.sin(2 * np.pi * X / 12) + 0.3 * np.cos(2 * np.pi * Y / 10)
panels = [(c, "粗尺度 $x_1$（~120 px）"), (m, "中尺度 $x_2$（~40 px）"),
          (f, "细尺度 $x_3$（~12 px）"), (c + m + f, "合成 $x=x_1+x_2+x_3$")]
fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.6), dpi=200)
for ax, (a, t) in zip(axes, panels):
    v = np.percentile(np.abs(a), 99)
    ax.imshow(a, cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_title(t, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#C9D9EA")
fig.suptitle("状态分解：$x=x_1+x_2+x_3$，三个分量共享同一套卫星/船载观测",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig_kf_scales_schematic.png", bbox_inches="tight")
plt.close(fig)

# ---------------- 图 2：谱域 KF 反演增益
G = 6.6743e-11
z0 = 5000.0
rho = 2200.0
lam = np.logspace(1, 3, 400)          # km
K = 2 * np.pi / (lam * 1e3)
A = 2 * np.pi * G * rho * np.exp(-K * z0) * 1e5
k0 = 2 * np.pi / 100e3
P0 = 2500.0 ** 2 * (k0 / np.maximum(K, 1e-9)) ** 2.5

fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.0), dpi=200)
ax = axes[0]
for ns, col in ((1.0, "#2E75B6"), (2.0, "#1F3864"), (4.0, "#C0392B")):
    g = (A ** 2 * P0) / (A ** 2 * P0 + ns ** 2)
    ax.semilogx(lam, g, color=col, lw=1.6, label=f"R$^{{1/2}}$={ns:.0f} mGal")
ax.axvspan(20, 200, color="#F2F5FA")
ax.set_xlabel("波长 (km)"); ax.set_ylabel("归一化增益 $A\\,K(k)$")
ax.set_ylim(0, 1.05)
ax.set_title("逐波数增益（归一化）：带内→1", fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
ax.loglog(lam, P0 * (A ** 2), color="#7F8C8D", lw=1.4, label="先验 $P_0A^2$")
ax.loglog(lam, np.full_like(lam, 4.0), color="#C0392B", lw=1.4, label="噪声 $R$")
ax.set_xlabel("波长 (km)"); ax.set_ylabel("方差 (mGal$^2$)")
ax.set_title("先验与噪声的交叉决定截止波长", fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.3, which="both")
fig.suptitle("谱域 KF 反演：$H_{\\rm post}(k)=K(k)\\,D(k)$，"
             "$K=P_0A/(A^2P_0+R)$", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig_kf_inversion_gain.png", bbox_inches="tight")
plt.close(fig)
print("saved")
