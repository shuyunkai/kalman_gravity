"""混合形式图：场图 + 功率谱 + 收敛曲线 + 密度散点。"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))

from three_method_compare import load_region_data
from aniso_kf_test import build

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, PALE, RED = "#0B2545", "#D6E2F0", "#C0392B"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 10})
LON, LAT = (132.0, 152.0), (26.0, 42.0)


def spectrum(f, dx, dy, nb=50):
    F = np.fft.fft2(f)
    P = np.abs(F) ** 2
    kx = 2 * np.pi * np.fft.fftfreq(f.shape[1], dx)
    ky = 2 * np.pi * np.fft.fftfreq(f.shape[0], dy)
    KX, KY = np.meshgrid(kx, ky)
    K = np.hypot(KX, KY).ravel()
    P = P.ravel()
    m = (K > 2 * np.pi / 500e3) & (K < 2 * np.pi / 5e3)
    k = K[m]; p = P[m]
    bins = np.logspace(np.log10(2*np.pi/500e3), np.log10(2*np.pi/5e3), nb)
    idx = np.digitize(k, bins)
    kk = []; pp = []
    for i in range(1, nb):
        q = idx == i
        if q.sum() > 5:
            kk.append(np.mean(k[q])); pp.append(np.mean(p[q]))
    kk = np.array(kk); pp = np.array(pp)
    return 2 * np.pi / kk / 1e3, pp


def add_spectrum(ax, ssv, ckf, dx, dy):
    l1, p1 = spectrum(ssv, dx, dy)
    l2, p2 = spectrum(ckf, dx, dy)
    lo = max(l1.min(), l2.min()); hi = min(l1.max(), l2.max())
    grid = np.logspace(np.log10(lo), np.log10(hi), 120)
    r = (np.interp(grid, l2[::-1], p2[::-1])
         / np.maximum(np.interp(grid, l1[::-1], p1[::-1]), 1e-30))
    ax.semilogx(grid, r, color=NAVY, lw=1.6)
    ax.axhline(1.0, color=RED, ls="--", lw=1)
    ax.set_xlabel("波长 (km)", fontsize=10)
    ax.set_ylabel("功率比（相干保持 KF / 1′）", fontsize=9)
    ax.set_title("短波能量增强", fontsize=9)
    ax.grid(alpha=0.2, which="both")
    ax.tick_params(labelsize=9)


def main():
    d = np.load(OUT / "validation_G_fields.npz")
    lon, lat = d["lon"], d["lat"]
    ssv, kf, ckf = d["ssv"], d["kf"], d["ckf"]
    val = np.asarray(d["val"], float)
    dx = (lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = (lat[1] - lat[0]) * 111320
    ext = [lon[0], lon[-1], lat[0], lat[-1]]
    k = 4
    sub = (slice(k, -k), slice(k, -k))
    v = np.nanpercentile(np.abs(ckf[sub]), 98)

    # 1) 产品页：全球 + 谱 + 卡片 + 柱状
    g = xr.open_dataset(ROOT / "data/icgem/grid/EGM2008_gravity_anomaly.nc")
    gv = g["gravity_anomaly"].values[::20, ::20]
    glon = g["lon"].values[::20]; glat = g["lat"].values[::20]
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 5.2), dpi=220)
    ax = axes[0, 0]
    im = ax.imshow(gv, cmap="RdBu_r", origin="upper",
                   extent=[glon[0], glon[-1], glat[-1], glat[0]],
                   vmin=-100, vmax=100, aspect="auto")
    ax.set_title("(a) 全球重力异常（EGM2008）", fontsize=12)
    ax.set_xlabel("经度 (°E)", fontsize=10); ax.set_ylabel("纬度 (°N)", fontsize=10)
    ax.tick_params(labelsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("mGal", fontsize=9); cb.ax.tick_params(labelsize=8)
    ax = axes[0, 1]
    add_spectrum(ax, ssv, ckf, dx, dy)
    ax.set_title("(b) 1′ 与 15″ 功率谱", fontsize=12)
    ax = axes[1, 0]; ax.axis("off")
    items = ["SSV33.1", "DTU21GRA", "SDUST2022GRA", "NSOAS24"]
    subs = ["1′ / 全球", "1′ / 全球", "1′ / 中国近海", "1′ / 全球"]
    for i, (name, subtxt) in enumerate(zip(items, subs)):
        y = 0.78 - i * 0.22
        ax.add_patch(plt.Rectangle((0.02, y), 0.96, 0.17, fc="#EEF3F9",
                                   ec="#0B2545", lw=1))
        ax.text(0.06, y + 0.085, name, va="center", fontsize=11, color="#0B2545")
        ax.text(0.62, y + 0.085, subtxt, va="center", fontsize=9, color="#5B6573")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("(c) 常用 1′ 产品", fontsize=12)
    ax = axes[1, 1]
    res = json.loads((OUT / "regions_coherence.json").read_text(encoding="utf-8"))
    names = ["南海-琉球", "菲律宾海", "日本"]
    keys = ["D_Nankai", "F_PhilSea", "G_Japan"]
    x = np.arange(3); w = 0.34
    ax.bar(x - w/2, [res[q]["ssv"] for q in keys], w, color=PALE, label="1′")
    ax.bar(x + w/2, [res[q]["ckf"] for q in keys], w, color=NAVY, label="相干保持 KF")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("RMSE (mGal)", fontsize=10)
    ax.set_title("(d) 三区留出精度", fontsize=12)
    ax.legend(fontsize=8, frameon=False); ax.tick_params(labelsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(OUT / "fig_new_products_mixed.png", bbox_inches="tight")
    plt.close(fig)

    # 2) 缺点页：场 + 谱 + 差值 + 分频
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 5.2), dpi=220)
    ax = axes[0, 0]
    im = ax.imshow(ssv[sub], cmap="RdBu_r", origin="lower", vmin=-v, vmax=v,
                   extent=ext, aspect="auto")
    ax.set_title("(a) 1′ 产品", fontsize=12)
    ax.set_xlabel("经度 (°E)", fontsize=10); ax.set_ylabel("纬度 (°N)", fontsize=10)
    ax.tick_params(labelsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("mGal", fontsize=9); cb.ax.tick_params(labelsize=8)
    ax = axes[0, 1]
    add_spectrum(ax, ssv, ckf, dx, dy)
    ax.set_title("(b) 功率谱：短波能量", fontsize=12)
    ax = axes[1, 0]
    diff = ckf - ssv
    vd = np.nanpercentile(np.abs(diff[sub]), 98)
    im = ax.imshow(diff[sub], cmap="PuOr_r", origin="lower", vmin=-vd, vmax=vd,
                   extent=ext, aspect="auto")
    ax.set_title("(c) 相干保持 KF − 1′", fontsize=12)
    ax.set_xlabel("经度 (°E)", fontsize=10); ax.set_ylabel("纬度 (°N)", fontsize=10)
    ax.tick_params(labelsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("mGal", fontsize=9); cb.ax.tick_params(labelsize=8)
    ax = axes[1, 1]
    bands = [(5e3, 10e3, "5–10"), (10e3, 20e3, "10–20"),
             (20e3, 50e3, "20–50")]
    from scale_decomposition import bandpass
    r1 = [float(np.std(bandpass(ssv, dx, dy, a, b))) for a, b, _ in bands]
    r2 = [float(np.std(bandpass(ckf, dx, dy, a, b))) for a, b, _ in bands]
    x = np.arange(3); w = 0.34
    ax.bar(x - w/2, r1, w, color=PALE, label="1′")
    ax.bar(x + w/2, r2, w, color=NAVY, label="相干保持 KF")
    ax.set_xticks(x); ax.set_xticklabels([b[2] for b in bands], fontsize=10)
    ax.set_ylabel("幅值 std (mGal)", fontsize=10)
    ax.set_title("(d) 分频幅值", fontsize=12)
    ax.legend(fontsize=9, frameon=False); ax.tick_params(labelsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(OUT / "fig_new_cons_mixed.png", bbox_inches="tight")
    plt.close(fig)

    # 3) 验证页：场 + 差值 + 密度散点 + 剖面统计
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 5.2), dpi=220)
    for ax, name, fld in ((axes[0, 0], "(a) 1′ 场", ssv),
                          (axes[0, 1], "(b) 相干保持 KF 场", ckf)):
        im = ax.imshow(fld[sub], cmap="RdBu_r", origin="lower", vmin=-v, vmax=v,
                       extent=ext, aspect="auto")
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("经度 (°E)", fontsize=10); ax.set_ylabel("纬度 (°N)", fontsize=10)
        ax.tick_params(labelsize=9)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("mGal", fontsize=9); cb.ax.tick_params(labelsize=8)
    ax = axes[1, 0]
    im = ax.imshow(diff[sub], cmap="PuOr_r", origin="lower", vmin=-vd, vmax=vd,
                   extent=ext, aspect="auto")
    ax.set_title("(c) 相干保持 KF − 1′", fontsize=12)
    ax.set_xlabel("经度 (°E)", fontsize=10); ax.set_ylabel("纬度 (°N)", fontsize=10)
    ax.tick_params(labelsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("mGal", fontsize=9); cb.ax.tick_params(labelsize=8)
    ax = axes[1, 1]
    ix = np.clip(np.searchsorted(lon, val[:, 0]), 0, len(lon)-1)
    iy = np.clip(np.searchsorted(lat, val[:, 1]), 0, len(lat)-1)
    pred = ckf[iy, ix]
    hb = ax.hexbin(val[:, 2], pred, gridsize=70, cmap="magma", mincnt=1, bins="log")
    lo, hi = np.percentile(val[:, 2], [1, 99])
    ax.plot([lo, hi], [lo, hi], "--", color="cyan", lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("船测重力 (mGal)", fontsize=10)
    ax.set_ylabel("相干保持 KF (mGal)", fontsize=10)
    ax.set_title("(d) 船测—模型密度散点", fontsize=12)
    ax.tick_params(labelsize=9)
    cb = fig.colorbar(hb, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("log10(点数)", fontsize=8); cb.ax.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fig_new_validation_mixed.png", bbox_inches="tight")
    plt.close(fig)

    # 4) 迭代页：输入/先验/收敛/终场/增量/散点
    data = load_region_data(LON, LAT, val_cruise="MR22-04_gravity_dmo", upscale=4)
    r = build(data["gravity_lr"].astype(float), data["bathy_15s"],
              data["lon_hr"], data["lat_hr"], w_keep=1.0, p0=2.0, n_iter=6,
              ship=data["ship_grav"], train=data["train_mask"],
              return_reconstructor=True)
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 5.2), dpi=220)
    for ax, name, fld in ((axes[0, 0], "(a) 先验：SSV + 地形", r.prior_2d),
                          (axes[0, 1], "(b) 观测：1′ + 船测", data["ssv_hr"]),
                          (axes[1, 0], "(d) 第 6 轮", r.x_history[5])):
        im = ax.imshow(fld[sub], cmap="RdBu_r", origin="lower", vmin=-v, vmax=v,
                       extent=ext, aspect="auto")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("经度 (°E)", fontsize=9); ax.set_ylabel("纬度 (°N)", fontsize=9)
        ax.tick_params(labelsize=8)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("mGal", fontsize=8); cb.ax.tick_params(labelsize=7)
    axes[0, 1].scatter(val[::6, 0], val[::6, 1], s=0.5, c="k", alpha=0.5)

    ax = axes[0, 2]
    chg = [float(np.sqrt(np.mean((r.x_history[i] - (r.prior_2d if i == 0 else r.x_history[i-1])) ** 2)))
           for i in range(len(r.x_history))]
    ax.semilogy(np.arange(1, len(chg)+1), chg, "o-", color=NAVY, label="每轮修正 RMS")
    ax.set_xlabel("迭代轮次", fontsize=9); ax.set_ylabel("mGal", fontsize=9)
    ax.set_title("(c) 收敛曲线", fontsize=11)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2); ax.tick_params(labelsize=8)

    ax = axes[1, 1]
    inc = r.x_history[5] - r.prior_2d
    vi = np.nanpercentile(np.abs(inc[sub]), 98)
    im = ax.imshow(inc[sub], cmap="PuOr_r", origin="lower", vmin=-vi, vmax=vi,
                   extent=ext, aspect="auto")
    ax.set_title("(e) 第 6 轮 − 初始先验", fontsize=11)
    ax.set_xlabel("经度 (°E)", fontsize=9); ax.set_ylabel("纬度 (°N)", fontsize=9)
    ax.tick_params(labelsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("mGal", fontsize=8); cb.ax.tick_params(labelsize=7)

    ax = axes[1, 2]
    ix = np.clip(np.searchsorted(lon, val[:, 0]), 0, len(lon)-1)
    iy = np.clip(np.searchsorted(lat, val[:, 1]), 0, len(lat)-1)
    ep = r.prior_2d[iy, ix] - val[:, 2]
    eq = r.x_history[5][iy, ix] - val[:, 2]
    rp = float(np.sqrt(np.mean(ep ** 2)))
    rq = float(np.sqrt(np.mean(eq ** 2)))
    bins = np.linspace(-20, 20, 41)
    ax.hist(ep, bins=bins, alpha=0.55, color="#90A4AE",
            label=f"初始先验  RMSE {rp:.2f}")
    ax.hist(eq, bins=bins, alpha=0.65, color=NAVY,
            label=f"第 6 轮  RMSE {rq:.2f}")
    ax.set_xlabel("船测残差 (mGal)", fontsize=9); ax.set_ylabel("点数", fontsize=9)
    ax.set_title("(f) 船测残差：初始先验 vs 第 6 轮", fontsize=10)
    ax.legend(fontsize=7, frameon=False); ax.grid(axis="y", alpha=0.2)
    ax.tick_params(labelsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_new_iterations_mixed.png", bbox_inches="tight")
    print("saved mixed figs")


if __name__ == "__main__":
    main()
