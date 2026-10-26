"""四条常用路线 + 四种缺点（不含深度学习/卡尔曼）。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).parent))
from scale_decomposition import bandpass

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, RED, GOLD, LIGHT = "#0B2545", "#C0392B", "#C9A227", "#EEF3F9"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})


def lanes():
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 5.6), dpi=220)
    data = [
        ("插值", ["1′ 格网", "核函数\n双线性 / 克里金", "逐点加密\n15″ 格点",
                  "15″ 格网\n无新增短波"],
         "$\\hat z(x_0)=\\sum w_i z_i$"),
        ("频域融合", ["测高重力\n+ 船测", "分带\n长波 / 短波", "波长域加权\n$W(k)$",
                      "15″ 场\n短波来自先验"],
         "$Z=WZ_{alt}+(1-W)Z_{ship}$"),
        ("最小二乘配置", ["观测 + 协方差", "协方差函数\n$C(r)$", "配置预测",
                          "15″ 场\n依赖协方差"],
         "$\\hat x=C_{xs}(C_{ss}+C_{nn})^{-1}z$"),
        ("多源迭代", ["初值 / 块平均", "残差\n$z-Hx$", "加权更新\n$W$", "15″ 场\n依赖初值"],
         "$x^{k+1}=x^k+W(z-Hx^k)$"),
    ]
    for ax, (name, steps, eq) in zip(axes.ravel(), data):
        ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                                    boxstyle="round,pad=0.01,rounding_size=0.03",
                                    fc="#F7F9FC", ec="#B0BEC5", lw=1))
        ax.add_patch(FancyBboxPatch((0.05, 0.82), 0.90, 0.13,
                                    boxstyle="round,pad=0.01,rounding_size=0.03",
                                    fc=NAVY, ec=NAVY))
        ax.text(0.5, 0.885, name, ha="center", va="center", fontsize=13,
                color=GOLD, fontweight="bold")
        for i, txt in enumerate(steps):
            y = 0.66 - i*0.16
            fc = "#FFF8E7" if i == 3 else "white"
            ax.add_patch(FancyBboxPatch((0.10, y), 0.80, 0.100,
                                        boxstyle="round,pad=0.01,rounding_size=0.02",
                                        fc=fc, ec=NAVY, lw=1))
            ax.text(0.5, y+0.050, txt, ha="center", va="center", fontsize=8.5)
            ax.text(0.055, y+0.050, str(i+1), ha="center", va="center",
                    fontsize=9, color=RED, fontweight="bold")
            if i < 3:
                ax.annotate("", xy=(0.5, y-0.055), xytext=(0.5, y-0.005),
                            arrowprops=dict(arrowstyle="-|>", color="#78909C",
                                            lw=1.2))
        ax.text(0.5, 0.025, eq, ha="center", fontsize=8, color=RED)
    fig.tight_layout()
    fig.savefig(OUT / "fig_new_methods_4lanes.png", bbox_inches="tight")
    plt.close(fig)


def cons():
    d = np.load(OUT / "validation_G_fields.npz")
    lon, lat, ssv = d["lon"], d["lat"], d["ssv"]
    ext = [lon[0], lon[-1], lat[0], lat[-1]]
    k = 4; sub = (slice(k, -k), slice(k, -k))
    dx = (lon[1]-lon[0])*111320*np.cos(np.radians(lat.mean()))
    dy = (lat[1]-lat[0])*111320

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 5.2), dpi=220)
    ax = axes[0, 0]
    i = np.argmin(np.abs(lat - 36.0))
    xs = lon[::4]; ys = ssv[i, ::4]
    ax.step(xs, ys, where="mid", color=NAVY, lw=1.4, label="1′ 格点")
    ax.plot(xs, ys, color=RED, lw=1.0, ls="--", label="插值")
    ax.set_xlim(141, 149)
    ax.set_title("(a) 插值：只加密，不增短波", fontsize=11)
    ax.set_xlabel("经度 (°E)", fontsize=8); ax.set_ylabel("mGal", fontsize=8)
    ax.legend(fontsize=8, frameon=False); ax.tick_params(labelsize=7)

    ax = axes[0, 1]
    pad = np.pad(ssv, 64, mode="reflect")
    bp_pad = bandpass(pad, dx, dy, 5e3, 20e3)[64:-64, 64:-64]
    bp = bandpass(ssv, dx, dy, 5e3, 20e3)
    edge = bp - bp_pad
    ve = np.nanpercentile(np.abs(edge), 98)
    im = ax.imshow(edge[sub], cmap="PuOr_r", origin="lower", vmin=-ve, vmax=ve,
                   extent=ext, aspect="auto")
    ax.set_title("(b) 频域融合：边界振铃", fontsize=11)
    ax.set_xlabel("经度 (°E)", fontsize=8); ax.set_ylabel("纬度 (°N)", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("mGal", fontsize=7)

    ax = axes[1, 0]
    r = np.linspace(0, 100, 200); a = 30
    ax.plot(r, np.exp(-r**2/(2*a**2)), color=NAVY, lw=1.6, label="高斯")
    ax.plot(r, np.exp(-r/a), color=RED, lw=1.6, ls="--", label="指数")
    ax.plot(r, (1+r/a)*np.exp(-r/a), color=GOLD, lw=1.6, ls=":", label="Matérn")
    ax.set_xlabel("距离 (km)", fontsize=8); ax.set_ylabel("协方差", fontsize=8)
    ax.set_title("(c) LSC：协方差函数选择敏感", fontsize=11)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2); ax.tick_params(labelsize=7)

    ax = axes[1, 1]
    ny, nx = ssv.shape
    blk = ssv.reshape(ny//4, 4, nx//4, 4).mean(axis=(1, 3))
    up = np.repeat(np.repeat(blk, 4, axis=0), 4, axis=1)
    res = ssv - up
    vr = np.nanpercentile(np.abs(res[sub]), 98)
    im = ax.imshow(res[sub], cmap="PuOr_r", origin="lower", vmin=-vr, vmax=vr,
                   extent=ext, aspect="auto")
    ax.set_title("(d) 迭代/块平均：块边界台阶", fontsize=11)
    ax.set_xlabel("经度 (°E)", fontsize=8); ax.set_ylabel("纬度 (°N)", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("mGal", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_new_cons_4methods.png", bbox_inches="tight")
    print("saved methods4/cons4")


if __name__ == "__main__":
    lanes()
    cons()
