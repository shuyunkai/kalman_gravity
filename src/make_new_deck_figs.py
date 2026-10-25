"""新叙事 13 页所需的配图。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from scale_decomposition import bandpass as kf_bandpass

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real" / "three_parts" / "newdeck"
OUT.mkdir(parents=True, exist_ok=True)
NAVY = "#0B2545"
MID = "#2E5C8A"
RED = "#C0392B"
GREY = "#5B6573"
PALE = "#D6E2F0"
LIGHT = "#EEF3F9"

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "font.size": 9,
})


def save(fig, name):
    p = OUT / name
    fig.savefig(p, bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("saved", p.name)


def load_japan():
    d = np.load(ROOT / "output_real" / "three_parts" / "ship_regions" / "G_Japan.npz")
    lon, lat = d["lon"], d["lat"]
    iy = (lat >= 32) & (lat <= 40); ix = (lon >= 141) & (lon <= 149)
    return lon[ix], lat[iy], d["ssv"][np.ix_(iy, ix)], d["kf"][np.ix_(iy, ix)], d["bathy"][np.ix_(iy, ix)]


def radial_spectrum(field, dx, dy, nb=36):
    f = field - np.nanmean(field)
    F = np.fft.fft2(np.nan_to_num(f))
    P = np.abs(F) ** 2
    kx = 2 * np.pi * np.fft.fftfreq(field.shape[1], dx)
    ky = 2 * np.pi * np.fft.fftfreq(field.shape[0], dy)
    KX, KY = np.meshgrid(kx, ky)
    K = np.hypot(KX, KY)
    bins = np.linspace(0, K.max(), nb)
    idx = np.digitize(K.ravel(), bins)
    pw = np.bincount(idx, weights=P.ravel(), minlength=nb + 1)
    cnt = np.bincount(idx, minlength=nb + 1)
    lam = 2 * np.pi / np.maximum(bins, 1e-12)
    return lam[1:], (pw[1:] / np.maximum(cnt[1:], 1))[1:]


# 1. 什么是地球重力场
def fig_concept():
    ds = xr.open_dataset(ROOT / "data" / "ssv32" / "grav_33.1.nc")
    z = ds["z"].values[::24, ::24].astype(float)
    lat = ds["lat"].values[::24]; lon = ds["lon"].values[::24]
    ds.close()
    fig = plt.figure(figsize=(13.5, 4.2), dpi=220)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    v = np.nanpercentile(np.abs(z), 98)
    im = ax.imshow(z, cmap="RdBu_r", vmin=-v, vmax=v, origin="lower",
                   extent=[lon[0], lon[-1], lat[0], lat[-1]], aspect="auto")
    ax.set_title("(a) 全球自由空气重力异常（SSV33.1，1′）", fontsize=10)
    ax.set_xlabel("经度 (°E)"); ax.set_ylabel("纬度 (°N)")
    fig.colorbar(im, ax=ax, shrink=0.82, pad=0.015, label="mGal")
    ax = fig.add_subplot(gs[0, 1])
    for txt, y in (("长波：地幔/地壳厚度", 0.78), ("中波：海山/盆地", 0.52),
                   ("短波：断裂/海底地形", 0.26)):
        ax.add_patch(FancyBboxPatch((0.06, y), 0.88, 0.16,
                                    boxstyle="round,pad=0.02,rounding_size=0.03",
                                    fc=LIGHT, ec=NAVY, lw=0.8))
        ax.text(0.5, y + 0.08, txt, ha="center", va="center", fontsize=10, color=NAVY)
    ax.annotate("", xy=(0.5, 0.94), xytext=(0.5, 0.84),
                arrowprops=dict(arrowstyle="-|>", color=GREY))
    ax.text(0.5, 0.03, "重力场 = 不同波长信号的叠加", ha="center", fontsize=10, color=RED)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("(b) 重力场的波长组成", fontsize=10)
    save(fig, "fig_new_concept.png")


# 2. 为什么重建
def fig_why():
    lon, lat, ssv, kf, bathy = load_japan()
    ext = [lon[0], lon[-1], lat[0], lat[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.9), dpi=220)
    v = np.percentile(np.abs(ssv), 98)
    axes[0].imshow(ssv, cmap="RdBu_r", vmin=-v, vmax=v, origin="lower", extent=ext, aspect="auto")
    axes[0].set_title("(a) 1′ 卫星重力", fontsize=10)
    axes[1].imshow(kf, cmap="RdBu_r", vmin=-v, vmax=v, origin="lower", extent=ext, aspect="auto")
    axes[1].set_title("(b) 15″ 重建重力", fontsize=10)
    d = kf - ssv
    vd = np.percentile(np.abs(d), 98)
    im = axes[2].imshow(d, cmap="PuOr_r", vmin=-vd, vmax=vd, origin="lower", extent=ext, aspect="auto")
    axes[2].set_title("(c) 15″ − 1′（短波增量）", fontsize=10)
    fig.colorbar(im, ax=axes[2], shrink=0.8, pad=0.02, label="mGal")
    for ax in axes:
        ax.set_xlabel("经度 (°E)"); ax.set_ylabel("纬度 (°N)")
        ax.tick_params(labelsize=7)
    fig.suptitle("同一海域：1′ 产品在 5–20 km 被平滑，15″ 场补回短波", fontsize=10, color=RED, y=1.02)
    save(fig, "fig_new_why.png")


# 3. 现有 1′ 产品
def fig_products():
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.2), dpi=220,
                             gridspec_kw={"width_ratios": [1.05, 1]})
    ax = axes[0]; ax.axis("off")
    cards = [
        ("SSV33.1", "1′ / 全球海洋", "Sandwell & Smith 系列，船测+测高"),
        ("DTU21GRA", "1′ / 全球海洋", "DTU 重力/垂线偏差产品"),
        ("SDUST2022GRA", "1′ / 中国近海", "中国近海高分辨率重力"),
        ("NSOAS24", "1′ / 全球海洋", "国家海洋卫星应用中心产品"),
    ]
    for i, (name, res, desc) in enumerate(cards):
        x = 0.03 + (i % 2) * 0.48; y = 0.52 - (i // 2) * 0.45
        ax.add_patch(FancyBboxPatch((x, y), 0.44, 0.38,
                                    boxstyle="round,pad=0.02,rounding_size=0.03",
                                    fc=LIGHT, ec=NAVY, lw=0.9))
        ax.text(x + 0.03, y + 0.28, name, fontsize=11, color=NAVY, fontweight="bold")
        ax.text(x + 0.03, y + 0.17, res, fontsize=9, color=RED)
        ax.text(x + 0.03, y + 0.05, desc, fontsize=7.5, color=GREY)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("(a) 常见 1′ 重力产品", fontsize=10)

    ax = axes[1]
    regions = ["南海-琉球", "菲律宾海", "日本综合区"]
    ssv = [5.344, 1.485, 3.960]; kf = [4.910, 1.480, 3.830]
    x = np.arange(3); w = 0.35
    ax.bar(x - w/2, ssv, w, color=PALE, label="1′ 产品")
    ax.bar(x + w/2, kf, w, color=NAVY, label="15″ 重建")
    for i in range(3):
        ax.text(x[i] - w/2, ssv[i] + 0.08, f"{ssv[i]:.2f}", ha="center", fontsize=8)
        ax.text(x[i] + w/2, kf[i] + 0.08, f"{kf[i]:.2f}", ha="center", fontsize=8)
        ax.text(x[i], max(ssv[i], kf[i]) + 0.55, f"↓{100*(1-kf[i]/ssv[i]):.1f}%",
                ha="center", fontsize=9, color=RED, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(regions)
    ax.set_ylabel("RMSE (mGal)"); ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(0, 6.6)
    ax.set_title("(b) 1′ 产品 vs 15″ 重建（留出船测）", fontsize=10, pad=10)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig_new_products.png")


# 4. 常用方法
def fig_methods():
    fig, ax = plt.subplots(figsize=(13.6, 4.0), dpi=220)
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    methods = [
        ("插值", "双线性/克里金\n把 1′ 直接加密", "平滑，短波是假的"),
        ("频域融合", "按波长融合\n测高+船测", "需要先验，边界效应"),
        ("深度学习", "成对样本\n学习重力→地形", "缺物理约束，样本少"),
        ("卡尔曼滤波", "状态方程+观测\n多源递推", "协方差难建模"),
    ]
    for i, (name, idea, cons) in enumerate(methods):
        x = 0.02 + i * 0.25
        ax.add_patch(FancyBboxPatch((x, 0.20), 0.19, 0.62,
                                    boxstyle="round,pad=0.02,rounding_size=0.03",
                                    fc=LIGHT if i < 3 else "#E8D5C4",
                                    ec=NAVY, lw=1.0))
        ax.text(x + 0.095, 0.72, name, ha="center", fontsize=12,
                color=NAVY, fontweight="bold")
        ax.text(x + 0.095, 0.52, idea, ha="center", fontsize=8.5, color=GREY)
        ax.text(x + 0.095, 0.30, cons, ha="center", fontsize=7.5, color=RED)
    ax.annotate("", xy=(0.5, 0.12), xytext=(0.5, 0.02),
                arrowprops=dict(arrowstyle="-|>", color=GREY))
    ax.text(0.5, 0.13, "共同目标：15″、5–20 km 可信", ha="center", fontsize=9, color=GREY)
    ax.set_title("常用 15″ 重力场重建方法", fontsize=11)
    save(fig, "fig_new_methods.png")


# 5. 缺点
def fig_cons():
    lon, lat, ssv, kf, bathy = load_japan()
    dx = (lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = (lat[1] - lat[0]) * 111320
    b_ssv = kf_bandpass(ssv, dx, dy, 5e3, 20e3)
    b_kf = kf_bandpass(kf, dx, dy, 5e3, 20e3)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.0), dpi=220)
    ax = axes[0]
    i = np.argmin(np.abs(lat - 36.0))
    ax.plot(lon, b_ssv[i], color=PALE, lw=2.2, label="1′ 产品")
    ax.plot(lon, b_kf[i], color=NAVY, lw=1.2, label="15″ 重建")
    ax.set_xlabel("经度 (°E)"); ax.set_ylabel("5–20 km 重力 (mGal)")
    ax.set_title("(a) 5–20 km 剖面：1′ 产品几乎无振幅", fontsize=10)
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1]
    bands = [(5e3, 10e3, "5–10"), (10e3, 20e3, "10–20"),
             (20e3, 50e3, "20–50"), (50e3, 100e3, "50–100")]
    r1 = [float(np.std(kf_bandpass(ssv, dx, dy, a, b))) for a, b, _ in bands]
    r2 = [float(np.std(kf_bandpass(kf, dx, dy, a, b))) for a, b, _ in bands]
    x = np.arange(len(bands)); w = 0.35
    ax.bar(x - w/2, r1, w, color=PALE, label="1′ 产品")
    ax.bar(x + w/2, r2, w, color=NAVY, label="15″ 重建")
    for i in range(len(bands)):
        ax.text(x[i]-w/2, r1[i]+0.05, f"{r1[i]:.2f}", ha="center", fontsize=7)
        ax.text(x[i]+w/2, r2[i]+0.05, f"{r2[i]:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([b[2] for b in bands])
    ax.set_xlabel("波长 (km)"); ax.set_ylabel("幅值 std (mGal)")
    ax.set_title("(b) 分频幅值：短波能量差异", fontsize=10)
    ax.legend(frameon=False, fontsize=8); ax.grid(axis="y", alpha=0.25)
    save(fig, "fig_new_cons.png")


# 6. 标准 KF
def fig_stdkf():
    fig, ax = plt.subplots(figsize=(13.0, 4.2), dpi=220)
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    boxes = [
        (0.03, 0.58, "预测", "$x^{-}=Ax$\n$P^{-}=APA^{T}+Q$"),
        (0.28, 0.58, "观测", "$z=Hx+v$\nR：观测噪声"),
        (0.53, 0.58, "增益", "$K=P^{-}H^{T}(HP^{-}H^{T}+R)^{-1}$"),
        (0.78, 0.58, "更新", "$x=x^{-}+K(z-Hx^{-})$\n$P=(I-KH)P^{-}$"),
    ]
    for x, y, name, eq in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 0.19, 0.26,
                                    boxstyle="round,pad=0.02,rounding_size=0.03",
                                    fc=LIGHT, ec=NAVY, lw=1.0))
        ax.text(x + 0.095, y + 0.19, name, ha="center", fontsize=11, color=NAVY, fontweight="bold")
        ax.text(x + 0.095, y + 0.08, eq, ha="center", fontsize=7.5, color=GREY)
    for x in (0.22, 0.47, 0.72):
        ax.annotate("", xy=(x + 0.06, 0.71), xytext=(x, 0.71),
                    arrowprops=dict(arrowstyle="-|>", color=GREY))
    ax.annotate("", xy=(0.875, 0.56), xytext=(0.875, 0.42),
                arrowprops=dict(arrowstyle="-|>", color=GREY))
    ax.text(0.875, 0.38, "下一轮", ha="center", fontsize=8, color=GREY)
    ax.text(0.5, 0.92, "标准卡尔曼滤波：预测—观测—增益—更新", ha="center",
            fontsize=12, color=NAVY, fontweight="bold")
    ax.text(0.5, 0.20, "优点：多源融合、权重由 R/Q 自动决定、给出不确定度",
            ha="center", fontsize=9, color=RED)
    save(fig, "fig_new_stdkf.png")


# 7. 标准 KF 问题
def fig_stdkf_problems():
    lon, lat, ssv, kf, bathy = load_japan()
    d = kf - ssv
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.9), dpi=220)
    ax = axes[0]
    xx, yy = np.meshgrid(np.arange(9), np.arange(7))
    ax.scatter(xx.ravel(), yy.ravel(), s=18, c=NAVY)
    ax.set_title("(a) 对角 P：格点被当作独立", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax = axes[1]
    ax.add_patch(Rectangle((0.1, 0.1), 0.8, 0.8, fill=False, ec=RED, lw=1.5))
    ax.scatter([0.5], [0.5], s=60, c=NAVY, zorder=3)
    ax.text(0.5, 0.93, "1′ 块平均 ≠ 点观测", ha="center", fontsize=9, color=RED)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("(b) 观测算子不匹配", fontsize=9)
    ax = axes[2]
    v = np.percentile(np.abs(d), 98)
    im = ax.imshow(d, cmap="PuOr_r", vmin=-v, vmax=v, origin="lower",
                   extent=[lon[0], lon[-1], lat[0], lat[-1]], aspect="auto")
    ax.set_title("(c) 船测条带/块边界", fontsize=9)
    ax.set_xlabel("经度 (°E)"); ax.set_ylabel("纬度 (°N)")
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="mGal")
    save(fig, "fig_new_stdkf_problems.png")


# 8. 本项目 KF
def fig_projectkf():
    fig, ax = plt.subplots(figsize=(13.2, 4.2), dpi=220)
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    items = [
        (0.03, "块观测", "H 取 1′ 块平均\n匹配卫星足迹"),
        (0.28, "空间相关", "P∘L 修正协方差\n相关长度 ≈13 km"),
        (0.53, "自适应", "新息估计 R_sat\n适应不同海域"),
        (0.78, "融合", "15″ 场 + 不确定度\nP 逐像元输出"),
    ]
    for x, name, txt in items:
        ax.add_patch(FancyBboxPatch((x, 0.30), 0.19, 0.46,
                                    boxstyle="round,pad=0.02,rounding_size=0.03",
                                    fc="#E8D5C4" if x > 0.7 else LIGHT,
                                    ec=NAVY, lw=1.0))
        ax.text(x + 0.095, 0.63, name, ha="center", fontsize=11, color=NAVY, fontweight="bold")
        ax.text(x + 0.095, 0.44, txt, ha="center", fontsize=8, color=GREY)
    for x in (0.22, 0.47, 0.72):
        ax.annotate("", xy=(x + 0.06, 0.53), xytext=(x, 0.53),
                    arrowprops=dict(arrowstyle="-|>", color=RED))
    ax.text(0.5, 0.88, "空间相关自适应卡尔曼滤波", ha="center",
            fontsize=13, color=NAVY, fontweight="bold")
    ax.text(0.5, 0.16,
            "$K=(P\\circ L)H^{T}[H(P\\circ L)H^{T}+R]^{-1}$",
            ha="center", fontsize=12, color=RED)
    save(fig, "fig_new_projectkf.png")


# 10. 验证
def fig_validation():
    regions = ["南海-琉球", "菲律宾海", "日本综合区"]
    ssv = [5.344, 1.485, 3.960]
    kf = [4.951, 1.480, 3.883]
    bla = [4.910, 1.547, 3.830]
    x = np.arange(3); w = 0.26
    fig, ax = plt.subplots(figsize=(12.8, 4.0), dpi=220)
    ax.bar(x - w, ssv, w, color=PALE, label="1′ 产品")
    ax.bar(x, kf, w, color=MID, label="标准 KF")
    ax.bar(x + w, bla, w, color=NAVY, label="本项目 KF")
    ann = ["最佳 ↓8.1%", "KF ↓0.3%", "最佳 ↓3.3%"]
    for i in range(3):
        for j, val in enumerate((ssv[i], kf[i], bla[i])):
            ax.text(x[i] + (j-1)*w, val + 0.05, f"{val:.2f}", ha="center", fontsize=7)
        ax.text(x[i], max(ssv[i], kf[i], bla[i]) + 0.45,
                ann[i], ha="center", fontsize=9,
                color=RED, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(regions)
    ax.set_ylabel("留出航次 RMSE (mGal)"); ax.set_ylim(0, 6.3)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("三个 15″ 区域的留出船测验证", fontsize=11)
    save(fig, "fig_new_validation.png")


# 11. 精度总结
def fig_accuracy():
    labels = ["南海-琉球", "菲律宾海", "日本综合区"]
    sat = [8.1, 0.3, 3.3]
    x = np.arange(3); w = 0.45
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 3.9), dpi=220,
                             gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    ax.bar(x, sat, w, color=NAVY, label="相对 1′ 产品")
    for i in range(3):
        ax.text(x[i], sat[i]+0.15, f"{sat[i]}%", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE 改善 (%)"); ax.set_ylim(0, 9.5)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("(a) 三个新增验证区的改善幅度", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax = axes[1]; ax.axis("off")
    cards = [("0.3–8.1%", "相对 1′ 产品"), ("3.59 / 4.84", "留出船测 mGal")]
    for i, (val, txt) in enumerate(cards):
        y = 0.72 - i * 0.31
        ax.add_patch(FancyBboxPatch((0.05, y), 0.9, 0.24,
                                    boxstyle="round,pad=0.02,rounding_size=0.03",
                                    fc=LIGHT, ec=NAVY, lw=0.8))
        ax.text(0.18, y+0.12, val, fontsize=13, color=RED, fontweight="bold", va="center")
        ax.text(0.55, y+0.12, txt, fontsize=9, color=GREY, va="center")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("(b) 关键精度", fontsize=10)
    save(fig, "fig_new_accuracy.png")


# 13. 总结
def fig_summary():
    fig, ax = plt.subplots(figsize=(12.8, 3.6), dpi=220)
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    cards = [("15″", "分辨率", "1.85 km 网格"),
             ("0.3–8.1%", "精度改善", "相对 1′ 产品"),
             ("61.6%", "断裂覆盖", "5 km，GM9")]
    for i, (big, mid, small) in enumerate(cards):
        x = 0.02 + i * 0.335
        ax.add_patch(FancyBboxPatch((x, 0.12), 0.29, 0.72,
                                    boxstyle="round,pad=0.02,rounding_size=0.04",
                                    fc=LIGHT, ec=NAVY, lw=1.0))
        ax.text(x+0.145, 0.62, big, ha="center", fontsize=20, color=RED, fontweight="bold")
        ax.text(x+0.145, 0.44, mid, ha="center", fontsize=12, color=NAVY, fontweight="bold")
        ax.text(x+0.145, 0.28, small, ha="center", fontsize=9, color=GREY)
    save(fig, "fig_new_summary.png")


if __name__ == "__main__":
    fig_concept()
    fig_why()
    fig_products()
    fig_methods()
    fig_cons()
    fig_stdkf()
    fig_stdkf_problems()
    fig_projectkf()
    fig_validation()
    fig_accuracy()
    fig_summary()
