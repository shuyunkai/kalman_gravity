"""应用场景页 + 1′ 场景表现页。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, PALE, RED, GOLD, LIGHT = "#0B2545", "#D6E2F0", "#C0392B", "#C9A227", "#EEF3F9"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})


def table(ax, rows, cols, widths, fs=8.5):
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    h = 1.0/(len(rows)+1); y = 1.0; x = 0
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
                    fontsize=fs-0.5, color="#1A1A1A")
            x += w
        y -= h


def scenarios():
    d = np.load(OUT / "validation_G_fields.npz")
    lon, lat = d["lon"], d["lat"]
    ckf = d["ckf"]
    bathy = np.load(ROOT / "output_real/three_parts/ship_regions/G_Japan.npz")["bathy"]
    dx = (lon[1]-lon[0])*111320*np.cos(np.radians(lat.mean()))
    dy = (lat[1]-lat[0])*111320
    ext = [lon[0], lon[-1], lat[0], lat[-1]]
    k = 4; sub = (slice(k, -k), slice(k, -k))
    from scale_decomposition import bandpass
    g5 = bandpass(ckf, dx, dy, 5e3, 20e3)
    gy, gx = np.gradient(g5, dy, dx); grad = np.hypot(gx, gy)
    boug = 0.0688 * bathy

    fig = plt.figure(figsize=(13.4, 5.2), dpi=220)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.3, 1.0], hspace=0.28,
                          wspace=0.10, width_ratios=[1, 1, 1, 1.15])
    ax = fig.add_subplot(gs[0, :3])
    rows = [
        ("大地测量", "geoid、垂线偏差、高程基准", ">100 / 20–100 km", ">100 km 稳定、局部精度"),
        ("重力匹配导航", "重力图匹配、惯导修正", "5–20 km 梯度", "连续 5–20 km 梯度"),
        ("海底地形/工程", "海山、断裂、陡坡", "5–20 km", "海山/断裂边界"),
        ("资源/深部", "盆地、基底、莫霍、地壳厚度", "布格、>100 km", ">100 km + 浅层细节"),
    ]
    table(ax, rows, ["应用", "典型用途", "依赖波段", "场景需要"],
          [0.15, 0.34, 0.22, 0.29], fs=8.5)

    af = fig.add_subplot(gs[0, 3]); af.axis("off")
    af.set_xlim(0, 1); af.set_ylim(0, 1)
    af.text(0.5, 0.96, "1′ 产品的重力来源", ha="center", fontsize=8.5,
            color=NAVY, fontweight="bold")
    prod_cols = ["产品", "重力来源"]
    prod_w = [0.43, 0.57]
    prod_rows = [
        ("SSV33.1", "卫星测高反演"),
        ("DTU21GRA", "多代卫星测高"),
        ("SDUST2022GRA", "雷达 + 激光测高"),
        ("NSOAS24", "多源测高融合"),
    ]
    x = 0.02; y = 0.82; hh = 0.10
    for c, w in zip(prod_cols, prod_w):
        af.add_patch(Rectangle((x, y), w, hh, fc=NAVY, ec="white", lw=0.8))
        af.text(x+w/2, y+hh/2, c, ha="center", va="center", color="white",
                fontsize=6.8, fontweight="bold")
        x += w
    for i, (p, s) in enumerate(prod_rows):
        yy = y - (i+1)*hh
        x = 0.02
        for val, w in zip((p, s), prod_w):
            af.add_patch(Rectangle((x, yy), w, hh, fc="#EEF3F9" if i % 2 == 0 else "white",
                                   ec="white", lw=0.8))
            af.text(x+0.01, yy+hh/2, val, ha="left", va="center", fontsize=6.2,
                    color="#1A1A1A")
            x += w
    af.text(0.5, 0.16, "海面高 → 重力异常 → 1′ 格网", ha="center",
            fontsize=6.2, color=RED)

    panels = [
        ("(a) 导航：5–20 km 梯度", grad*1000, "Reds", None, "mGal/km"),
        ("(b) 地形/工程：5–20 km", g5, "RdBu_r", "sym", "mGal"),
        ("(c) 资源：布格异常", boug, "RdBu_r", "sym", "mGal"),
    ]
    for ax, (title, fld, cmap, mode, unit) in zip(
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
             fig.add_subplot(gs[1, 2:])], panels):
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
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(unit, fontsize=6.5); cb.ax.tick_params(labelsize=6)
    fig.savefig(OUT / "fig_new_apps_scenarios.png", bbox_inches="tight")
    plt.close(fig)


def gap():
    fig = plt.figure(figsize=(13.4, 5.0), dpi=220)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 1.0], hspace=0.18)
    ax = fig.add_subplot(gs[0])
    rows = [
        ("大地测量", ">100 / 20–100 km", "基本满足", ">100/20–100 km 保留"),
        ("重力匹配导航", "5–20 km", "不足", "20 km 以下被平滑，短波梯度缺失"),
        ("海底地形/工程", "5–20 km", "不足", "海山/断裂边界模糊，陡坡细节丢失"),
        ("资源/深部", "布格、>100 km", ">100 km 满足", "浅部细节仍缺"),
    ]
    table(ax, rows, ["场景", "依赖波段", "1′ 表现", "具体表现"],
          [0.18, 0.22, 0.13, 0.47], fs=9)

    ax = fig.add_subplot(gs[1])
    ax.set_xlim(1, 100); ax.set_ylim(0, 1); ax.axis("off")
    ax.axvspan(5, 20, color=RED, alpha=0.15)
    ax.plot([1, 100], [0.55, 0.55], color=NAVY, lw=1.2)
    ax.text(10, 0.75, "5–20 km 场景无法满足", ha="center", fontsize=10,
            color=RED, fontweight="bold")
    ax.annotate("", xy=(60, 0.55), xytext=(25, 0.55),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.text(70, 0.55, "需要 15″ 网格重建", fontsize=9, color=NAVY,
            va="center", fontweight="bold")
    ax.text(10, 0.30, "目标短波", ha="center", fontsize=8, color=RED)
    fig.savefig(OUT / "fig_new_1min_scenarios.png", bbox_inches="tight")
    plt.close(fig)


scenarios()
gap()
print("saved")
