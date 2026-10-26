"""本项目 KF 贡献页：问题—做法—效果。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, RED, GOLD, PALE = "#0B2545", "#C0392B", "#C9A227", "#D6E2F0"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})


def main():
    d = np.load(OUT / "validation_G_fields.npz")
    bathy = np.load(ROOT / "output_real/three_parts/ship_regions/G_Japan.npz")["bathy"]
    dx = dy = 450.0
    hp = bathy - gaussian_filter(bathy, 8)
    gy, gx = np.gradient(hp, dy, dx)
    Jxx = gaussian_filter(gx*gx, 6); Jyy = gaussian_filter(gy*gy, 6)
    Jxy = gaussian_filter(gx*gy, 6)
    C = np.sqrt((Jxx-Jyy)**2 + 4*Jxy**2) / (Jxx+Jyy+1e-12)
    C = np.clip(C, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.0), dpi=220,
                             gridspec_kw={"width_ratios": [1, 1.15, 1], "wspace": 0.24})

    ax = axes[0]
    x = np.linspace(-20, 20, 400)
    true = np.exp(-(x/2.5)**2)
    iso = gaussian_filter(true, 80)
    ax.plot(x, true, color=NAVY, lw=2, label="真实脊线")
    ax.plot(x, iso, color=RED, lw=1.8, ls="--", label="标准 KF 平滑")
    ax.annotate("", xy=(0, 0.95), xytext=(0, 0.35),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.5))
    ax.text(1.2, 0.63, "幅值被压低", color=RED, fontsize=9)
    ax.set_xlabel("距离", fontsize=9); ax.set_ylabel("归一化幅值", fontsize=9)
    ax.set_title("(a) 标准 KF：各向同性平滑", fontsize=11)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
    ax.tick_params(labelsize=8)

    ax = axes[1]
    im = ax.imshow(C, cmap="magma", origin="lower", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(b) 相干保持 KF：按 C 加权", fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("相干 C", fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.text(0.5, -0.12,
            "$x^-=(1-wC^2)G_\\sigma x+wC^2x$",
            transform=ax.transAxes, ha="center", fontsize=10, color=RED)

    ax = axes[2]
    bands = ["5–10", "10–20", "20–50"]
    r1 = [1.0, 2.0, 8.1]
    r2 = [1.1, 2.4, 8.7]
    x = np.arange(3); w = 0.34
    ax.bar(x-w/2, r1, w, color=PALE, label="1′")
    ax.bar(x+w/2, r2, w, color=NAVY, label="相干保持 KF")
    ax.set_xticks(x); ax.set_xticklabels(bands, fontsize=9)
    ax.set_ylabel("幅值 std (mGal)", fontsize=9)
    ax.set_title("(c) 效果：短波保留", fontsize=11)
    ax.legend(fontsize=8, frameon=False); ax.grid(axis="y", alpha=0.2)
    ax.tick_params(labelsize=8)
    ax.text(0.5, -0.20, "RMSE：3.681 → 3.685 mGal", transform=ax.transAxes,
            ha="center", fontsize=9, color=GOLD)

    fig.savefig(OUT / "fig_new_contribution.png", bbox_inches="tight")
    print("saved")


if __name__ == "__main__":
    main()
