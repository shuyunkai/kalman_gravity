"""生成"杰青风格"汇报 PPT 用的 5 张图（低文字密度、信息靠图）。

配色纪律（杰青规范：全篇不超过 3 种主色）:
    深蓝 #1F4E79  |  中蓝 #5C8AA8  |  浅蓝 #9DB4C0   +  灰 #7F7F7F   +  强调深红 #C00000
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np
import xarray as xr
from scipy.ndimage import zoom
from scipy.interpolate import RegularGridInterpolator

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.linewidth": 0.9,
    "savefig.facecolor": "white",
})

# academic-navy 品牌值
NAVY, MID, LIGHT, GREY, RED = "#16365C", "#4A7AA7", "#D8E0EC", "#5B6573", "#C0392B"
INK = "#1C2430"
# 图示节点淡彩组（仅用于流程/架构节点）
NODE = {
    "blue":    ("#DCE6F5", "#9DB6DC", "#16365C"),
    "green":   ("#DCEBDC", "#9CC79B", INK),
    "purple":  ("#E4DEF0", "#B7A9D6", INK),
    "yellow":  ("#FBF1CE", "#E0C97A", INK),
    "pink":    ("#F8DEE2", "#E3A7B0", INK),
    "neutral": ("#EEF1F5", "#CAD2DE", INK),
}


def node(ax, x, y, w, h, kind, title, lines=(), rx=0.06, fs=11.0, ts=13.5):
    """淡彩节点：淡底 + 深一档细描边 + navy/墨色文字。"""
    fill, edge, txt = NODE[kind]
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={rx}",
                                   fc=fill, ec=edge, lw=1.5, zorder=3,
                                   clip_on=False))
    ax.text(x + w / 2, y + h - 0.30, title, ha="center", va="center",
            fontsize=ts, fontweight="bold", color=txt, zorder=4)
    for i, s in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.72 - i * 0.30, s, ha="center",
                va="center", fontsize=fs, color=txt, zorder=4)
OUT = Path(r"D:\gravity\ppt_jq\figs")
OUT.mkdir(parents=True, exist_ok=True)
GLOB = Path(r"D:\gravity\kalman_gravity\output_global")
SSV = Path(r"D:\gravity\kalman_gravity\data\ssv32\grav_33.1.nc")


# ---------------------------------------------------------------- 0. 什么是卡尔曼滤波
def fig_kf_cycle():
    """卡尔曼滤波的「预测—加权校正」递推循环，并标出各符号在本项目中的对应物。"""
    fig, ax = plt.subplots(figsize=(11.0, 4.75), dpi=200)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 11.0); ax.set_ylim(0, 4.75); ax.axis("off")

    def box(x, y, w, h, fc, title, lines, tc="white"):
        ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                       fc=fc, ec="none", zorder=3))
        ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center",
                fontsize=13.5, fontweight="bold", color=tc, zorder=4)
        for i, t in enumerate(lines):
            ax.text(x + w / 2, y + h - 0.76 - i * 0.30, t, ha="center",
                    va="center", fontsize=10.5, color=tc, zorder=4)

    node(ax, 0.15, 2.70, 4.05, 1.62, "blue", "① 预测：把状态向量推进一步",
         ["$x^{-}=Fx$        $P^{-}=FPF^{T}+Q$",
          "用空间相关模型外推，不确定度随之增大"])
    node(ax, 6.80, 2.70, 4.05, 1.62, "purple", "② 校正：用观测向量修正",
         ["$x=x^{-}+K(z-Hx^{-})$", "$P=(I-KH)P^{-}$"])
    node(ax, 3.40, 1.15, 4.20, 1.20, "yellow", "卡尔曼增益 K")
    ax.text(5.50, 1.67, "$K=P^{-}H^{T}(HP^{-}H^{T}+R)^{-1}$", ha="center",
            va="center", fontsize=13, fontweight="bold", color=INK, zorder=4)
    ax.text(5.50, 1.33, "谁更可信就多听谁的：观测越准，K 越大",
            ha="center", va="center", fontsize=10.5, color=INK, zorder=4)

    # 预测 → 校正
    ax.annotate("", xy=(6.62, 3.51), xytext=(4.42, 3.51),
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color=NAVY,
                                shrinkA=0, shrinkB=0))
    ax.text(5.51, 3.76, "预测值 $x^{-}$, $P^{-}$", ha="center", fontsize=10.5, color=NAVY)
    # 递推回环：沿底部绕行，不穿过任何框
    yl = 0.68
    ax.plot([8.82, 8.82, 2.18], [2.66, yl, yl], color=RED, lw=1.8, zorder=2,
            solid_capstyle="round")
    ax.annotate("", xy=(2.18, 2.66), xytext=(2.18, yl),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color=RED))
    ax.text(5.50, 0.36, "校正结果成为下一步的预测输入 —— 逐像元递推",
            ha="center", fontsize=11, color=RED, fontweight="bold")
    ax.text(5.50, 0.06, "直到全区域收敛：这正是 15″ 场能从 1′ 观测里被解出来的原因",
            ha="center", fontsize=10.5, color=GREY)

    ax.text(0.15, 4.62, "放到本问题里：x = 状态向量（15″ 重力场）　F = 状态转移矩阵（空间平滑）　"
                        "H = 观测矩阵（4×4 块平均）　z = 观测向量（卫星 1′ 与船测）",
            fontsize=10.5, color="#12314a", va="center")
    fig.savefig(OUT / "fig_kf_cycle.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 1. 是什么
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(12, 4.6), dpi=200)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 12.25); ax.set_ylim(0, 4.6); ax.axis("off")

    def box(x, y, w, h, title, sub, kind, fs=12, subfs=9.5):
        fill, edge, txt = NODE[kind]
        ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10",
                                       fc=fill, ec=edge, lw=1.5, clip_on=False))
        ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center",
                color=txt, fontsize=fs, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                color=txt, fontsize=subfs, linespacing=1.5)

    box(0.10, 2.55, 2.75, 1.35, "卫星测高重力", "1′ 网格 · 覆盖全球\n短波被海水衰减", "neutral")
    box(0.10, 1.15, 2.75, 1.15, "船载重力", "稀疏测线 · 精度最高\n点约束", "neutral")
    box(0.10, 0.05, 2.75, 0.90, "海底地形", "15″ 全覆盖\n物理先验", "neutral")

    ax.text(3.02, 2.0, "+", ha="center", va="center", fontsize=26, color=GREY)
    ax.text(3.02, 0.62, "+", ha="center", va="center", fontsize=26, color=GREY)

    box(3.45, 0.95, 2.85, 2.60, "卡尔曼滤波", "贝叶斯最优融合\n误差显式建模\n逐像元递推", "purple", fs=14)

    for y in (3.22, 1.72, 0.50):
        ax.annotate("", xy=(3.24, 2.25), xytext=(3.07, y),
                    arrowprops=dict(arrowstyle="-|>", lw=1.4, color=GREY,
                                    shrinkA=0, shrinkB=0))
    ax.annotate("", xy=(7.15, 2.25), xytext=(6.52, 2.25),
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color=NAVY,
                                shrinkA=0, shrinkB=0))

    box(7.35, 0.95, 2.70, 2.60, "15″ 重力场", "分辨率提升 4 倍\n逐像元不确定度\n来源可追溯", "green", fs=14)

    box(10.25, 2.60, 1.65, 0.95, "精度↑", "1′ → 15″", "blue", fs=12, subfs=9)
    box(10.25, 1.50, 1.65, 0.95, "成本↓", "少跑船", "blue", fs=12, subfs=9)
    box(10.25, 0.40, 1.65, 0.95, "可复现", "带版本溯源", "blue", fs=12, subfs=9)
    for y in (3.07, 1.97, 0.87):
        ax.annotate("", xy=(10.16, y), xytext=(10.06, y),
                    arrowprops=dict(arrowstyle="-|>", lw=1.2, color=GREY,
                                    shrinkA=0, shrinkB=0))
    ax.text(6.0, 4.30, "", ha="center")
    fig.savefig(OUT / "fig1_pipeline.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 2. 为什么需要
def fig_gap():
    ds = xr.open_dataset(GLOB / "kalman_gravity_15s_v2.nc", engine="h5netcdf")
    d1 = xr.open_dataset(SSV, engine="h5netcdf")
    i0 = int((35.0 + 80) * 240); j0 = int((140.5 + 180) * 240)
    n = 240
    hi = ds["gravity_anomaly"].isel(lat=slice(i0, i0 + n),
                                    lon=slice(j0, j0 + n)).values.astype(float)
    lo = d1["z"].isel(lat=slice(i0 // 4, i0 // 4 + n // 4),
                      lon=slice(j0 // 4, j0 // 4 + n // 4)).values.astype(float)
    lat = ds.lat.values[i0:i0 + n]; lon = ds.lon.values[j0:j0 + n]
    ext = [lon[0], lon[-1], lat[0], lat[-1]]
    vmin, vmax = np.nanpercentile(hi, [2, 98])

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), dpi=200,
                             gridspec_kw=dict(width_ratios=[1, 1, 1.35], wspace=0.30))
    for ax, z, ttl in zip(axes[:2], [lo, hi],
                          ["① 卫星重力模型：1′ 网格", "② 本方法：15″ 结果"]):
        im = ax.imshow(z, extent=ext, origin="lower", cmap="Blues_r",
                       vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")
        ax.set_title(ttl, fontsize=11.5, color=NAVY, fontweight="bold", pad=7)
        ax.set_xlabel("经度 (°E)", fontsize=9.5)
        ax.set_xticks([140.75, 141.0, 141.25])
        ax.tick_params(labelsize=8.5)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=8)
        ax.axhline(35.45, color=RED, lw=1.2, ls="--")
    axes[0].set_ylabel("纬度 (°N)", fontsize=9.5)

    # 右: 沿 35.45°N 的剖面 —— 直观说明"1′ 是台阶, 15″ 才有细节"
    ax = axes[2]
    k = int(np.argmin(np.abs(lat - 35.45)))
    ax.step(lon, np.repeat(lo[k // 4], 4), where="mid", color=GREY, lw=1.6,
            label="卫星模型 1′")
    ax.plot(lon, hi[k], color=NAVY, lw=1.9, label="本方法 15″")
    ax.set_title("③ 沿 35.45°N 的剖面：细节被还原", fontsize=11.5,
                 color=NAVY, fontweight="bold", pad=7)
    ax.set_xlabel("经度 (°E)", fontsize=9.5); ax.set_ylabel("重力异常 (mGal)", fontsize=9.5)
    ax.set_xticks([140.75, 141.0, 141.25]); ax.tick_params(labelsize=8.5)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    ax.grid(ls=":", lw=0.6, color="0.85"); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.annotate("海沟轴部", xy=(lon[np.argmin(hi[k])], hi[k].min()),
                xytext=(lon[np.argmin(hi[k])] + 0.16, hi[k].min() + 22),
                fontsize=9.5, color=RED,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2))

    fig.savefig(OUT / "fig2_gap.png", bbox_inches="tight")
    plt.close(fig); ds.close(); d1.close()


# ---------------------------------------------------------------- 3. 为什么科学
def fig_why():
    fig, ax = plt.subplots(figsize=(12.6, 4.5), dpi=200)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 36); ax.set_ylim(0, 11); ax.axis("off")

    def pill(x, y, w, h, txt, fc, fs=12, tc="white"):
        ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                       fc=fc, ec="none"))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                color=tc, fontsize=fs, fontweight="bold")

    # ---- 第 1 列: 观测矩阵 ----
    for i in range(4):                        # 粗网格 1′
        for j in range(4):
            ax.add_patch(mp.Rectangle((1.2 + 2.0 * j, 5.8 + 1.05 * i), 2.0, 1.05,
                                      fc=LIGHT, ec="white", lw=1.2))
    for i in range(8):                        # 细网格 15″
        for j in range(8):
            ax.add_patch(mp.Rectangle((1.2 + 1.0 * j, 1.4 + 0.55 * i), 1.0, 0.55,
                                      fc="white", ec=MID, lw=0.6))
    ax.add_patch(mp.Rectangle((1.2, 5.8), 8.0, 4.2, fc="none", ec=NAVY, lw=2.2))
    ax.annotate("", xy=(5.2, 5.55), xytext=(5.2, 5.75),
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color=NAVY))
    ax.text(5.2, 10.45, "① 观测方程写得出", ha="center", fontsize=13,
            color=NAVY, fontweight="bold")
    ax.text(9.7, 7.0, "1′ 观测向量\n=\n15″ 状态向量\n块平均\n+ 噪声", ha="left", va="center",
            fontsize=10.5, color=NAVY, linespacing=1.5)
    ax.text(5.2, 0.55, "每个观测对应明确的物理量，不是经验加权",
            ha="center", fontsize=9.5, color=GREY)

    # ---- 第 2 列: 误差建模 ----
    ax.text(19.0, 10.45, "② 误差来自数据本身", ha="center", fontsize=13,
            color=NAVY, fontweight="bold")
    for k, (nm, val, c) in enumerate([("卫星重力", "3.0 mGal", MID),
                                      ("船载重力", "1.0 mGal", NAVY),
                                      ("过程噪声", "2.0 mGal", LIGHT)]):
        y = 7.6 - k * 2.05
        pill(13.4, y, 11.2, 1.45, "", c)
        ax.text(14.1, y + 0.72, nm, fontsize=11.5, color="white",
                va="center", fontweight="bold")
        ax.text(24.0, y + 0.72, val, fontsize=11.5, color="white",
                va="center", ha="right")
    ax.text(19.0, 0.55, "全部来自数据的精度指标，可写进论文、可被复算",
            ha="center", fontsize=9.5, color=GREY)

    # ---- 第 3 列: 可证伪 ----
    ax.text(30.3, 10.45, "③ 设计上可被证伪", ha="center", fontsize=13,
            color=NAVY, fontweight="bold")
    pill(26.0, 7.0, 8.6, 1.6, "参与解算的航次", NAVY, fs=11.5)
    pill(26.0, 4.4, 8.6, 1.6, "整条留出的航次", LIGHT, fs=11.5, tc=NAVY)
    ax.annotate("", xy=(30.3, 6.1), xytext=(30.3, 6.9),
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color=GREY))
    ax.text(30.3, 2.6, "用没见过的航次打分", ha="center", fontsize=11,
            color=RED, fontweight="bold")
    ax.text(30.3, 0.55, "不留验证数据，就无从证明精度", ha="center",
            fontsize=9.5, color=GREY)

    ax.plot([12.2, 12.2], [1.2, 9.8], color="0.88", lw=1.0)
    ax.plot([24.6, 24.6], [1.2, 9.8], color="0.88", lw=1.0)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(OUT / "fig3_why.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 4. 证据
def fig_multiregion():
    regions = ["日本海沟北段", "本州东岸", "南海-琉球", "菲律宾海中部", "综合验证区"]
    methods = ["传统插值", "频域融合", "本方法"]
    rmse = np.array([[6.187, 6.020, 5.797], [7.043, 6.699, 6.406],
                     [5.252, 5.152, 5.080], [1.607, 1.563, 1.565],
                     [4.674, 4.575, 4.448]])
    n = [9695, 3762, 12310, 6459, 108103]
    fig, ax = plt.subplots(figsize=(11.6, 4.5), dpi=200)
    x = np.arange(len(regions)); w = 0.26
    for i, (m, c) in enumerate(zip(methods, [LIGHT, MID, NAVY])):
        b = ax.bar(x + (i - 1) * w, rmse[:, i], w, label=m, color=c,
                   edgecolor="white", linewidth=0.8, zorder=3)
        for xi, v in zip(b, rmse[:, i]):
            ax.text(xi.get_x() + xi.get_width() / 2, v + 0.07, f"{v:.2f}",
                    ha="center", fontsize=9)
    for j in range(len(regions)):
        ax.text(x[j], max(rmse[j]) + 0.62,
                f"{100*(1-rmse[j,2]/rmse[j,0]):+.1f}%", ha="center",
                fontsize=10.5, color=RED, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\n{n_:,} 点" for r, n_ in zip(regions, n)], fontsize=10)
    ax.set_ylabel("与独立航次的偏差 RMSE (mGal)", fontsize=10.5)
    ax.set_ylim(0, 8.8)
    ax.legend(frameon=False, fontsize=10.5, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.13))
    ax.grid(axis="y", ls=":", lw=0.6, color="0.8", zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(1.0, -0.235, "红色为相对传统插值的改善；每个区域都用未参与解算的整条航次打分",
            transform=ax.transAxes, ha="right", fontsize=9, color=GREY)
    fig.savefig(OUT / "fig4_multiregion.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 5. 优势
def fig_advantage():
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 5.6), dpi=200)
    fig.subplots_adjust(hspace=0.42, wspace=0.22)

    ax = axes[0, 0]
    labs = ["1′", "15″"]
    ax.bar(labs, [1, 4], color=[LIGHT, NAVY], width=0.5, zorder=3)
    ax.set_title("① 分辨率提升 4 倍", fontsize=12, color=NAVY, fontweight="bold")
    ax.set_ylabel("每度网格数（相对）", fontsize=9.5)
    ax.text(1, 4.18, "15″ ≈ 460 m", ha="center", fontsize=9.5, color=NAVY)
    ax.text(0, 1.18, "1′ ≈ 1.85 km", ha="center", fontsize=9.5, color=GREY)

    ax = axes[0, 1]
    ax.bar(["传统插值", "频域融合", "本方法"], [4.674, 4.575, 4.448],
           color=[LIGHT, MID, NAVY], width=0.55, zorder=3)
    ax.set_ylim(4.2, 4.85)
    ax.set_title("② 更准：综合验证区 RMSE", fontsize=12, color=NAVY, fontweight="bold")
    ax.set_ylabel("RMSE (mGal)", fontsize=9.5)
    for i, v in enumerate([4.674, 4.575, 4.448]):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=9.5)
    ax.annotate("−4.8%", xy=(1.86, 4.49), xytext=(0.95, 4.76), fontsize=11,
                color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4))

    ax = axes[1, 0]
    regs = ["日本\n海沟", "本州\n东岸", "南海\n琉球", "菲律宾\n海中部", "综合\n验证区"]
    imp = [6.3, 9.0, 3.3, 2.6, 4.8]
    ax.bar(regs, imp, color=NAVY, width=0.55, zorder=3)
    ax.set_title("③ 能推广：参数不改，跨海域都有效", fontsize=12,
                 color=NAVY, fontweight="bold")
    ax.set_ylabel("相对传统插值改善 (%)", fontsize=9.5)
    for i, v in enumerate(imp):
        ax.text(i, v + 0.25, f"+{v}%", ha="center", fontsize=9.5, color=NAVY)

    ax = axes[1, 1]; ax.axis("off")
    ax.set_title("④ 可复现：结果自带版本指纹", fontsize=12, color=NAVY,
                 fontweight="bold")
    ax.add_patch(mp.FancyBboxPatch((0.02, 0.06), 0.96, 0.80,
                                   boxstyle="round,pad=0.02", fc="#F2F5F8",
                                   ec=MID, transform=ax.transAxes))
    ax.text(0.06, 0.70, "code_commit   = 1b64f6f…", fontsize=9.5,
            family="monospace", color=NAVY, transform=ax.transAxes)
    ax.text(0.06, 0.53, "worktree_clean = yes", fontsize=9.5,
            family="monospace", color=NAVY, transform=ax.transAxes)
    ax.text(0.06, 0.36, "src_sha256    = f83a814…", fontsize=9.5,
            family="monospace", color=NAVY, transform=ax.transAxes)
    ax.text(0.06, 0.19, "source 掩码：标记哪些像元是解算、哪些是外推", fontsize=9.5,
            color=NAVY, transform=ax.transAxes)

    for ax in axes.ravel()[:3]:
        ax.grid(axis="y", ls=":", lw=0.6, color="0.85", zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.savefig(OUT / "fig5_advantage.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 6. 两种口径
def fig_scales():
    """同一份重建结果在两种评估口径下的 RMSE（本项目自己的验证运行）。"""
    regions = ["综合验证区\n108 103 点", "南海-琉球\n12 310 点", "菲律宾海中部\n6 459 点"]
    rmse_15s = np.array([4.173, 4.748, 1.455])   # 15″ 点对船载原始 10 s 点
    rmse_1m = np.array([5.207, 5.425, 1.474])    # 1′ 块均值对块均值
    fig, ax = plt.subplots(figsize=(10.4, 4.0), dpi=200)
    x = np.arange(len(regions))
    w = 0.32
    b1 = ax.bar(x - w / 2, rmse_15s, w, label="15″ 点口径", color=NAVY,
                edgecolor="white", zorder=3)
    b2 = ax.bar(x + w / 2, rmse_1m, w, label="1′ 块口径", color=LIGHT,
                edgecolor="white", zorder=3)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.08,
                f"{b.get_height():.2f}", ha="center", fontsize=10)
    for j in range(len(regions)):
        ax.annotate("", xy=(x[j] + w / 2, rmse_1m[j] - 0.10),
                    xytext=(x[j] - w / 2, rmse_15s[j] + 0.32),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2,
                                    connectionstyle="arc3,rad=-0.25"))
        ax.text(x[j], max(rmse_1m[j], rmse_15s[j]) + 0.55,
                f"+{rmse_1m[j]-rmse_15s[j]:.2f}", ha="center",
                fontsize=10, color=RED, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(regions, fontsize=10.5)
    ax.set_ylabel("与独立航次比较的 RMSE (mGal)", fontsize=10.5)
    ax.set_ylim(0, 7.4)
    ax.legend(frameon=False, fontsize=10.5, ncol=2, loc="upper left")
    ax.grid(axis="y", ls=":", lw=0.6, color="0.85", zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(OUT / "fig6_scales.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # 只生成 PPT 实际使用的三张图，避免产生冗余文件
    fig_pipeline(); print("fig1_pipeline ok")
    fig_gap();      print("fig2_gap ok")
    fig_scales();   print("fig6_scales ok")
    print("输出:", OUT)
