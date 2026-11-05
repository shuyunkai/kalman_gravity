"""标准 KF 问题页 2×2。"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})


def main():
    d = np.load(OUT / "validation_G_fields.npz")
    lon, lat, ssv, ckf = d["lon"], d["lat"], d["ssv"], d["ckf"]
    bathy = np.load(ROOT / "output_real/three_parts/ship_regions/G_Japan.npz")["bathy"]
    ext = [lon[0], lon[-1], lat[0], lat[-1]]
    k = 4; sub = (slice(k, -k), slice(k, -k))

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 5.2), dpi=220)
    ax = axes[0, 0]
    gy, gx = np.mgrid[0:9, 0:9]
    ax.scatter(gx, gy, s=45, c="#0B2545")
    ax.set_xlim(-0.5, 8.5); ax.set_ylim(-0.5, 8.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(a) 对角 P：格点被当作独立", fontsize=11)
    ax.set_aspect("equal")

    ax = axes[0, 1]
    ax.add_patch(plt.Rectangle((0.08, 0.08), 0.84, 0.84, fill=False,
                               ec="#C0392B", lw=1.8))
    for i in range(4):
        for j in range(4):
            ax.scatter([0.20 + i*0.20], [0.20 + j*0.20], s=30, c="#9FB3C8")
    ax.scatter([0.5], [0.5], s=70, c="#0B2545", zorder=5)
    ax.text(0.5, 0.02, "1′ 块内 16 个 15″ 像素，只给一个观测值", ha="center",
            fontsize=8, color="#C0392B")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("(b) 1′ 块平均 ≠ 点观测", fontsize=11)

    ax = axes[1, 0]
    ex = [139.25, 139.95, 34.55, 35.15]
    bx = (lon >= ex[0]) & (lon <= ex[1]); by = (lat >= ex[2]) & (lat <= ex[3])
    z = bathy[np.ix_(by, bx)]
    z = z[:z.shape[0]//4*4, :z.shape[1]//4*4]
    blk = z.reshape(z.shape[0]//4, 4, z.shape[1]//4, 4).mean(axis=(1, 3))
    up = np.repeat(np.repeat(blk, 4, axis=0), 4, axis=1)
    ls = LightSource(azdeg=315, altdeg=45)
    extc = [lon[bx][0], lon[bx][z.shape[1]-1], lat[by][0], lat[by][z.shape[0]-1]]
    ax.imshow(ls.shade(up, cmap=plt.cm.Greys, vert_exag=0.6, blend_mode="overlay"),
              extent=extc, origin="lower", aspect="auto")
    dxs = lon[bx][1]-lon[bx][0]; dys = lat[by][1]-lat[by][0]
    for xx in lon[bx][:z.shape[1]:4]:
        ax.axvline(xx, color="#FFFFFF", lw=0.35, alpha=0.55)
    for yy in lat[by][:z.shape[0]:4]:
        ax.axhline(yy, color="#FFFFFF", lw=0.35, alpha=0.55)
    i0 = (z.shape[1]//8)//4*4; j0 = (z.shape[0]//2)//4*4
    ax.add_patch(plt.Rectangle((lon[bx][i0], lat[by][j0]), 4*dxs, 4*dys,
                               fill=False, ec="#C0392B", lw=1.6))
    ax.set_title("(c) 块平均地形：块边界台阶", fontsize=11)
    ax.set_xlabel("经度 (°E)", fontsize=9); ax.set_ylabel("纬度 (°N)", fontsize=9)
    ax.tick_params(labelsize=7)

    ax = axes[1, 1]
    x = np.linspace(-10, 10, 400)
    spike = np.exp(-(x / 0.45) ** 2)
    smooth = np.exp(-(x / 4.0) ** 2)
    ax.plot(x, spike, color="#C0392B", lw=1.8, label="标准 KF：单点")
    ax.plot(x, smooth, color="#0B2545", lw=1.8, ls="--", label="邻域修正")
    ax.axvline(0, color="#9E9E9E", lw=0.8)
    ax.set_xlabel("离船测点距离（像元）", fontsize=9)
    ax.set_ylabel("归一化修正量", fontsize=9, labelpad=8)
    ax.set_title("(d) 单点更新 vs 邻域修正", fontsize=11)
    ax.legend(fontsize=8, frameon=False); ax.tick_params(labelsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    p = OUT / "fig_new_stdkf_problems_2x2.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    main()
