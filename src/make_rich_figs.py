"""验证页/精度页的多要素大图。"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, MID, PALE, RED = "#0B2545", "#2E5C8A", "#D6E2F0", "#C0392B"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False,
                     "font.size": 7})


def load():
    d = np.load(OUT / "validation_G_fields.npz")
    return (d["lon"], d["lat"], d["ssv"], d["kf"], d["ckf"],
            np.asarray(d["val"], float))


def sample(field, lon, lat, pts):
    ix = np.clip(np.searchsorted(lon, pts[:, 0]), 0, len(lon) - 1)
    iy = np.clip(np.searchsorted(lat, pts[:, 1]), 0, len(lat) - 1)
    return field[iy, ix]


def main():
    lon, lat, ssv, kf, ckf, val = load()
    ext = [lon[0], lon[-1], lat[0], lat[-1]]
    k = 4
    sub = (slice(k, -k), slice(k, -k))
    v = np.nanpercentile(np.abs(ckf[sub]), 98)

    fig = plt.figure(figsize=(14.4, 4.9), dpi=220)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.92], hspace=0.38, wspace=0.16)

    for j, (name, fld) in enumerate((("(a) 1′ 产品", ssv),
                                     ("(b) 标准 KF", kf),
                                     ("(c) 相干保持 KF", ckf))):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(fld[sub], cmap="RdBu_r", origin="lower", vmin=-v, vmax=v,
                       extent=ext, aspect="auto")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("经度 (°E)", fontsize=7)
        if j == 0:
            ax.set_ylabel("纬度 (°N)", fontsize=7)
        ax.tick_params(labelsize=6)
    cax = fig.add_axes([0.912, 0.56, 0.008, 0.34])
    fig.colorbar(im, cax=cax, label="mGal")

    ax = fig.add_subplot(gs[1, 0])
    diff = ckf - ssv
    vd = np.nanpercentile(np.abs(diff[sub]), 98)
    im2 = ax.imshow(diff[sub], cmap="PuOr_r", origin="lower", vmin=-vd, vmax=vd,
                    extent=ext, aspect="auto")
    ax.set_title("(d) 相干保持 KF − 1′ 差值", fontsize=9)
    ax.set_xlabel("经度 (°E)", fontsize=7); ax.set_ylabel("纬度 (°N)", fontsize=7)
    ax.tick_params(labelsize=6)
    fig.colorbar(im2, ax=ax, fraction=0.03, pad=0.01)

    ax = fig.add_subplot(gs[1, 1])
    err = np.abs(sample(ckf, lon, lat, val) - val[:, 2])
    sc = ax.scatter(val[:, 0], val[:, 1], c=err, s=1.2, cmap="magma_r",
                    vmin=0, vmax=np.nanpercentile(err, 95), linewidths=0)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_title("(e) 留出船测点误差分布", fontsize=9)
    ax.set_xlabel("经度 (°E)", fontsize=7); ax.set_ylabel("纬度 (°N)", fontsize=7)
    ax.tick_params(labelsize=6)
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("|误差| (mGal)", fontsize=6); cb.ax.tick_params(labelsize=5)

    ax = fig.add_subplot(gs[1, 2])
    res = json.loads((OUT / "regions_coherence.json").read_text(encoding="utf-8"))
    names = ["南海-琉球", "菲律宾海", "日本"]
    keys = ["D_Nankai", "F_PhilSea", "G_Japan"]
    x = np.arange(3); w = 0.26
    ax.bar(x - w, [res[k]["ssv"] for k in keys], w, color=PALE, label="1′")
    ax.bar(x, [res[k]["kf"] for k in keys], w, color=MID, label="标准 KF")
    ax.bar(x + w, [res[k]["ckf"] for k in keys], w, color=NAVY, label="相干保持 KF")
    for i, k in enumerate(keys):
        ax.text(x[i] + w, res[k]["ckf"] + 0.08, f"{res[k]['ckf']:.2f}",
                ha="center", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7)
    ax.set_title("(f) 三区留出 RMSE", fontsize=9)
    ax.set_ylabel("mGal", fontsize=7); ax.set_ylim(0, 6.2)
    ax.legend(fontsize=6, frameon=False); ax.tick_params(labelsize=6)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(OUT / "fig_new_validation_rich.png", bbox_inches="tight")
    plt.close(fig)

    # 精度页：改善柱状 + 误差分布 + CDF
    err_s = sample(ssv, lon, lat, val) - val[:, 2]
    err_k = sample(kf, lon, lat, val) - val[:, 2]
    err_c = sample(ckf, lon, lat, val) - val[:, 2]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.7), dpi=220,
                             gridspec_kw={"width_ratios": [1, 1.05, 1.05]})
    ax = axes[0]
    imp = [8.7, 0.2, 6.3]
    ax.bar(names, imp, color=NAVY, width=0.55)
    for i, vv in enumerate(imp):
        ax.text(i, vv + 0.2, f"{vv}%", ha="center", fontsize=8)
    ax.set_ylabel("RMSE 改善 (%)", fontsize=7); ax.set_ylim(0, 10)
    ax.set_title("(a) 相对 1′ 产品", fontsize=9)
    ax.tick_params(labelsize=7); ax.grid(axis="y", alpha=0.2)

    ax = axes[1]
    bins = np.linspace(-8, 8, 60)
    for e, lab, col in ((err_s, "1′", PALE), (err_k, "标准 KF", MID),
                        (err_c, "相干保持 KF", NAVY)):
        ax.hist(e, bins=bins, density=True, histtype="step", lw=1.3,
                color=col, label=lab)
    ax.set_xlabel("误差 (mGal)", fontsize=7); ax.set_ylabel("密度", fontsize=7)
    ax.set_title("(b) 留出误差分布", fontsize=9)
    ax.legend(fontsize=6, frameon=False); ax.tick_params(labelsize=6)

    ax = axes[2]
    for e, lab, col in ((err_s, "1′", PALE), (err_k, "标准 KF", MID),
                        (err_c, "相干保持 KF", NAVY)):
        ax.plot(np.sort(np.abs(e)), np.linspace(0, 1, len(e)), color=col,
                lw=1.4, label=lab)
    ax.set_xlim(0, 8); ax.set_xlabel("|误差| (mGal)", fontsize=7)
    ax.set_ylabel("累积比例", fontsize=7)
    ax.set_title("(c) 误差累积分布", fontsize=9)
    ax.legend(fontsize=6, frameon=False); ax.tick_params(labelsize=6)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_new_accuracy_rich.png", bbox_inches="tight")
    print("saved rich figs")


if __name__ == "__main__":
    main()
