"""重做第 2/3/4/5 页配图，保证图与页面主题一致。"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).parent))
from scale_decomposition import bandpass

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, PALE, RED, GOLD = "#0B2545", "#D6E2F0", "#C0392B", "#C9A227"
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


def main():
    d = np.load(OUT / "validation_G_fields.npz")
    lon, lat, ssv, ckf = d["lon"], d["lat"], d["ssv"], d["ckf"]
    d2 = np.load(ROOT / "output_real/three_parts/ship_regions/G_Japan.npz")
    bathy = d2["bathy"]
    val = np.asarray(d["val"], float)
    dx = (lon[1]-lon[0])*111320*np.cos(np.radians(lat.mean()))
    dy = (lat[1]-lat[0])*111320
    ext = [lon[0], lon[-1], lat[0], lat[-1]]
    k = 4; sub = (slice(k, -k), slice(k, -k))

    # ---------- 第 2 页：应用 ----------
    g5 = bandpass(ckf, dx, dy, 5e3, 20e3)
    gy, gx = np.gradient(g5, dy, dx)
    grad = np.hypot(gx, gy)
    boug = 0.0688 * bathy
    deep = bandpass(ckf, dx, dy, 100e3, 500e3)

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 5.2), dpi=220)
    items = [
        ("(a) 重力辅助导航：短波梯度", grad, "Reds", None),
        ("(b) 海底地形与工程：5–20 km 短波", g5, "RdBu_r", "sym"),
        ("(c) 资源勘查：布格异常", boug, "RdBu_r", "sym"),
        ("(d) 深部结构：长波重力", deep, "RdBu_r", "sym"),
    ]
    for ax, (title, fld, cmap, mode) in zip(axes.ravel(), items):
        z = fld[sub]
        if mode == "sym":
            v = np.nanpercentile(np.abs(z), 98)
            im = ax.imshow(z, cmap=cmap, origin="lower", vmin=-v, vmax=v,
                           extent=ext, aspect="auto")
        else:
            v = np.nanpercentile(z, 98)
            im = ax.imshow(z, cmap=cmap, origin="lower", vmin=0, vmax=v,
                           extent=ext, aspect="auto")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("经度 (°E)", fontsize=8); ax.set_ylabel("纬度 (°N)", fontsize=8)
        ax.tick_params(labelsize=7)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("mGal" if mode == "sym" else "mGal/m", fontsize=7)
    axes[0, 0].plot([139.5, 141.0, 143.0], [34.0, 35.2, 36.5],
                    color=GOLD, lw=2)
    axes[0, 0].scatter([143.0], [36.5], s=30, c=GOLD)
    fig.tight_layout()
    fig.savefig(OUT / "fig_new_why_apps.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- 第 3 页：1′ 产品 ----------
    g = xr.open_dataset(ROOT / "data/icgem/grid/EGM2008_gravity_anomaly.nc")
    gv = g["gravity_anomaly"].values[::20, ::20]
    glon = g["lon"].values[::20]; glat = g["lat"].values[::20]

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 5.2), dpi=220)
    ax = axes[0, 0]
    im = ax.imshow(gv, cmap="RdBu_r", origin="upper",
                   extent=[glon[0], glon[-1], glat[-1], glat[0]],
                   vmin=-100, vmax=100, aspect="auto")
    ax.set_title("(a) 全球 1′ 重力产品（EGM2008）", fontsize=11)
    ax.set_xlabel("经度 (°E)", fontsize=8); ax.set_ylabel("纬度 (°N)", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("mGal", fontsize=7)

    ax = axes[0, 1]
    ax.set_xlim(-0.4, 4.4); ax.set_ylim(-1.0, 4.6); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 4, 4, fill=False, ec=NAVY, lw=2.2))
    for i in range(4):
        for j in range(4):
            ax.add_patch(Rectangle((i, j), 1, 1, fc="#EEF3F9" if (i+j) % 2 else "#D6E2F0",
                                   ec="#90A4AE", lw=0.6))
    ax.set_title("(b) 1′ 格网 vs 15″ 格网", fontsize=11)
    ax.text(2, 4.25, "1′ 格网（约 1.85 km）", ha="center", fontsize=10, color=NAVY)
    ax.text(2, -0.55, "每个 1′ 格网含 4×4 个 15″ 格网（约 0.46 km）",
            ha="center", fontsize=8, color="#5B6573")

    ax = axes[1, 0]; ax.axis("off")
    items2 = [("SSV33.1", "1′ / 全球"), ("DTU21GRA", "1′ / 全球"),
              ("SDUST2022GRA", "1′ / 中国近海"), ("NSOAS24", "1′ / 全球")]
    for i, (name, sub2) in enumerate(items2):
        y = 0.78 - i*0.22
        ax.add_patch(Rectangle((0.02, y), 0.96, 0.17, fc="#EEF3F9", ec=NAVY, lw=1))
        ax.text(0.06, y+0.085, name, va="center", fontsize=11, color=NAVY)
        ax.text(0.60, y+0.085, sub2, va="center", fontsize=9, color="#5B6573")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("(c) 四类常用 1′ 产品", fontsize=11)

    ax = axes[1, 1]
    lam, p = spectrum(ssv, dx, dy)
    ax.loglog(lam, p/p.max(), color=NAVY, lw=1.4)
    ax.axvline(20, color=RED, ls="--", lw=1)
    ax.text(22, 0.2, "20 km", color=RED, fontsize=8)
    ax.set_xlabel("波长 (km)", fontsize=8); ax.set_ylabel("归一化功率", fontsize=8)
    ax.set_title("(d) 1′ 产品的能量集中在长波", fontsize=11)
    ax.grid(alpha=0.2, which="both"); ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_new_products_page.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- 第 4 页：三种方法流程 ----------
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.0), dpi=220)
    flows = [
        ("插值", ["1′ 格网", "插值核\n双线性 / 克里金", "15″ 格网"]),
        ("频域融合", ["测高重力", "按波长加权\n长波 + 短波", "15″ 场"]),
        ("深度学习", ["成对样本", "网络训练\n重力→地形", "预测 15″"]),
    ]
    for ax, (name, steps) in zip(axes, flows):
        ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.5, 0.96, name, ha="center", fontsize=14, color=NAVY,
                fontweight="bold")
        for i, txt in enumerate(steps):
            y = 0.70 - i*0.30
            ax.add_patch(FancyBboxPatch((0.12, y), 0.76, 0.20,
                                        boxstyle="round,pad=0.015,rounding_size=0.03",
                                        fc="#EEF3F9", ec=NAVY, lw=1))
            ax.text(0.5, y+0.10, txt, ha="center", va="center", fontsize=10,
                    color=NAVY)
            if i < 2:
                ax.annotate("", xy=(0.5, y-0.10), xytext=(0.5, y-0.03),
                            arrowprops=dict(arrowstyle="-|>", color="#5B6573", lw=1.2))
    fig.tight_layout()
    fig.savefig(OUT / "fig_new_methods_flows.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- 第 5 页：三种方法的缺点 ----------
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 5.2), dpi=220)
    # (a) 1′ 阶梯剖面 vs 插值
    ax = axes[0, 0]
    i = np.argmin(np.abs(lat - 36.0))
    ax.step(lon, ssv[i], where="mid", color=NAVY, lw=1.3, label="1′ 产品")
    ax.plot(lon, ssv[i], color=RED, lw=0.8, ls="--", label="插值 15″")
    ax.set_xlim(141, 149)
    ax.set_title("(a) 插值：只加密，不增短波", fontsize=11)
    ax.set_xlabel("经度 (°E)", fontsize=8); ax.set_ylabel("mGal", fontsize=8)
    ax.legend(fontsize=8, frameon=False); ax.tick_params(labelsize=7)

    # (b) 频域融合边界振铃
    ax = axes[0, 1]
    pad = np.pad(ssv, 64, mode="reflect")
    bp_pad = bandpass(pad, dx, dy, 5e3, 20e3)[64:-64, 64:-64]
    bp = bandpass(ssv, dx, dy, 5e3, 20e3)
    edge = bp - bp_pad
    ve = np.nanpercentile(np.abs(edge), 98)
    im = ax.imshow(edge, cmap="PuOr_r", origin="lower", vmin=-ve, vmax=ve,
                   extent=ext, aspect="auto")
    ax.set_title("(b) 频域融合：边界振铃", fontsize=11)
    ax.set_xlabel("经度 (°E)", fontsize=8); ax.set_ylabel("纬度 (°N)", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("mGal", fontsize=7)

    # (c) 深度学习样本示意
    ax = axes[1, 0]
    n = np.linspace(10, 1000, 100)
    err = 8 / np.sqrt(n) + 0.8
    ax.semilogx(n, err, color=NAVY, lw=1.8)
    ax.set_xlabel("训练样本数", fontsize=8); ax.set_ylabel("外推误差（示意）", fontsize=8)
    ax.set_title("(c) 深度学习：样本少、外推弱", fontsize=11)
    ax.grid(alpha=0.2); ax.tick_params(labelsize=7)

    # (d) 共同问题：谱
    ax = axes[1, 1]
    lam, p = spectrum(ssv, dx, dy)
    ax.loglog(lam, p/p.max(), color=NAVY, lw=1.4)
    ax.axvline(20, color=RED, ls="--", lw=1)
    ax.set_xlabel("波长 (km)", fontsize=8); ax.set_ylabel("归一化功率", fontsize=8)
    ax.set_title("(d) 三种方法都缺 20 km 以下能量", fontsize=11)
    ax.grid(alpha=0.2, which="both"); ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_new_cons_methods.png", bbox_inches="tight")
    print("saved pages 2-5 figs")


if __name__ == "__main__":
    main()
