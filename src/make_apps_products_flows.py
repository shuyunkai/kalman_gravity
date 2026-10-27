"""第 2/3/4 页：应用矩阵、1′ 产品能力表、三种方法泳道图。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).parent))
from scale_decomposition import bandpass

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, PALE, RED, GOLD, LIGHT = "#0B2545", "#D6E2F0", "#C0392B", "#C9A227", "#EEF3F9"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})


def table(ax, rows, cols, widths, header=True, fs=9):
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    y = 1.0
    h = 1.0 / (len(rows) + (1 if header else 0))
    if header:
        x = 0
        for c, w in zip(cols, widths):
            ax.add_patch(Rectangle((x, y-h), w, h, fc=NAVY, ec="white", lw=1))
            ax.text(x+w/2, y-h/2, c, ha="center", va="center", color="white",
                    fontsize=fs, fontweight="bold")
            x += w
        y -= h
    for r in rows:
        x = 0
        for val, w in zip(r, widths):
            ax.add_patch(Rectangle((x, y-h), w, h, fc=LIGHT, ec="white", lw=1))
            ax.text(x+0.012, y-h/2, val, ha="left", va="center",
                    fontsize=fs-0.5, color="#1A1A1A", wrap=True)
            x += w
        y -= h


def applications():
    d = np.load(OUT / "validation_G_fields.npz")
    lon, lat, ckf, bathy = d["lon"], d["lat"], d["ckf"], np.load(
        ROOT / "output_real/three_parts/ship_regions/G_Japan.npz")["bathy"]
    dx = (lon[1]-lon[0])*111320*np.cos(np.radians(lat.mean()))
    dy = (lat[1]-lat[0])*111320
    ext = [lon[0], lon[-1], lat[0], lat[-1]]
    k = 4; sub = (slice(k, -k), slice(k, -k))
    g5 = bandpass(ckf, dx, dy, 5e3, 20e3)
    gy, gx = np.gradient(g5, dy, dx); grad = np.hypot(gx, gy)
    deep = bandpass(ckf, dx, dy, 100e3, 500e3)
    boug = 0.0688 * bathy

    fig = plt.figure(figsize=(13.4, 5.2), dpi=220)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.28, wspace=0.12)

    ax = fig.add_subplot(gs[0, :])
    rows = [
        ("大地测量", "geoid、垂线偏差、高程基准", "长波 + 中波", "基本满足", "提升局部精度"),
        ("重力匹配导航", "重力图匹配、惯导修正", "5–20 km 短波梯度", "不足", "提供可匹配短波"),
        ("海底地形/工程", "海山、断裂、陡坡、管线选线", "5–20 km", "不足", "恢复短波结构"),
        ("资源/深部", "盆地、基底、莫霍、地壳厚度", "布格/长波", "长波满足", "浅部细节仍缺"),
    ]
    table(ax, rows, ["应用", "典型用途", "依赖波段", "1′ 能力", "15″ 作用"],
          [0.13, 0.31, 0.20, 0.14, 0.22], fs=9)

    panels = [
        ("(a) 导航：短波梯度", grad, "Reds", None),
        ("(b) 地形/工程：5–20 km", g5, "RdBu_r", "sym"),
        ("(c) 资源：布格异常", boug, "RdBu_r", "sym"),
    ]
    for ax, (title, fld, cmap, mode) in zip(
            [fig.add_subplot(gs[1, i]) for i in range(3)], panels):
        z = fld[sub]
        if mode == "sym":
            v = np.nanpercentile(np.abs(z), 98)
            im = ax.imshow(z, cmap=cmap, origin="lower", vmin=-v, vmax=v,
                           extent=ext, aspect="auto")
        else:
            v = np.nanpercentile(z, 98)
            im = ax.imshow(z, cmap=cmap, origin="lower", vmin=0, vmax=v,
                           extent=ext, aspect="auto")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(OUT / "fig_new_why_matrix.png", bbox_inches="tight")
    plt.close(fig)


def products():
    fig = plt.figure(figsize=(13.4, 5.2), dpi=220)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0],
                          height_ratios=[1.15, 1.0], hspace=0.26, wspace=0.10)
    ax = fig.add_subplot(gs[0, :])
    rows = [
        ("SSV33.1", "测高重力（全球）", "全球海洋", "1′ 格网，有效约 12–20 km", "20 km 以下被平滑"),
        ("DTU21GRA", "测高重力（全球）", "全球海洋", "1′ 格网", "近岸/极区精度下降"),
        ("SDUST2022GRA", "雷达 + 激光测高（全球）", "80°S–82°N", "1′ 格网", "激光测高覆盖有限"),
        ("NSOAS24", "全球海洋重力产品", "80°S–80°N", "1′ 格网", "北界仅 80°N"),
    ]
    table(ax, rows, ["产品", "来源/方法", "覆盖", "有效分辨率", "主要不足/说明"],
          [0.13, 0.28, 0.13, 0.24, 0.22], fs=9)

    ax = fig.add_subplot(gs[1, 0])
    ax.set_xlim(-0.4, 4.4); ax.set_ylim(-1.0, 4.6); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 4, 4, fill=False, ec=NAVY, lw=2.2))
    for i in range(4):
        for j in range(4):
            ax.add_patch(Rectangle((i, j), 1, 1,
                                   fc="#EEF3F9" if (i+j) % 2 else "#D6E2F0",
                                   ec="#90A4AE", lw=0.6))
    ax.set_title("(a) 1′ 格网包含 4×4 个 15″ 格网", fontsize=10)
    ax.text(2, -0.55, "格网加密容易，但短波不会自动出现", ha="center",
            fontsize=8, color=RED)

    ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.02, 0.94, "(b) 1′ 产品不能满足的场景", fontsize=10, color=NAVY,
            fontweight="bold")
    items = ["5–20 km 海山、断裂、陡坡", "重力匹配导航的短波梯度",
             "近岸、浅水、极区", "管线/电缆选线", "局部资源与浅层构造"]
    for i, t in enumerate(items):
        y = 0.76 - i*0.15
        ax.text(0.04, y, "•", fontsize=12, color=RED)
        ax.text(0.10, y, t, fontsize=9, color="#1A1A1A", va="center")
    fig.savefig(OUT / "fig_new_products_matrix.png", bbox_inches="tight")
    plt.close(fig)


def methods():
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.2), dpi=220,
                             gridspec_kw={"wspace": 0.12})
    lanes = [
        ("插值", ["1′ 格网", "选择核\n双线性 / 克里金", "逐点加密\n15″ 格点", "15″ 格网\n无新增短波"],
         "$\\hat z(x_0)=\\sum w_i z_i$"),
        ("频域融合", ["测高重力\n+ 船测", "分带\n长波 / 短波", "波长域加权\n$W(k)$", "15″ 场\n短波来自先验"],
         "$Z(k)=W(k)Z_{alt}(k)+(1-W)Z_{ship}(k)$"),
        ("深度学习", ["成对样本\n重力—地形", "网络\nCNN / U-Net", "训练\n损失 + 验证", "预测 15″\n依赖样本覆盖"],
         "$\\hat y=f_\\theta(x)$"),
    ]
    for ax, (name, steps, eq) in zip(axes, lanes):
        ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.add_patch(FancyBboxPatch((0.03, 0.03), 0.94, 0.94,
                                    boxstyle="round,pad=0.01,rounding_size=0.03",
                                    fc="#F7F9FC", ec="#B0BEC5", lw=1))
        ax.add_patch(FancyBboxPatch((0.06, 0.86), 0.88, 0.10,
                                    boxstyle="round,pad=0.01,rounding_size=0.03",
                                    fc=NAVY, ec=NAVY))
        ax.text(0.5, 0.91, name, ha="center", va="center", fontsize=14,
                color=GOLD, fontweight="bold")
        for i, txt in enumerate(steps):
            y = 0.70 - i*0.18
            fc = "#FFF8E7" if i == 3 else "white"
            ax.add_patch(FancyBboxPatch((0.10, y), 0.80, 0.13,
                                        boxstyle="round,pad=0.01,rounding_size=0.02",
                                        fc=fc, ec=NAVY, lw=1))
            ax.text(0.5, y+0.065, txt, ha="center", va="center", fontsize=9)
            ax.text(0.055, y+0.065, str(i+1), ha="center", va="center",
                    fontsize=9, color=RED, fontweight="bold")
            if i < 3:
                ax.annotate("", xy=(0.5, y-0.055), xytext=(0.5, y-0.005),
                            arrowprops=dict(arrowstyle="-|>", color="#78909C",
                                            lw=1.2, connectionstyle="arc3,rad=0.0"))
        ax.text(0.5, 0.045, eq, ha="center", fontsize=9, color=RED)
    fig.savefig(OUT / "fig_new_methods_lanes.png", bbox_inches="tight")
    plt.close(fig)
    print("saved apps/products/methods")


if __name__ == "__main__":
    applications()
    products()
    methods()
