"""补充三张图：结果对比图 / 下一步路线图 / 封面研究区图。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "savefig.facecolor": "white",
})
NAVY, MID, LIGHT, GREY, RED = "#16365C", "#4A7AA7", "#D8E0EC", "#5B6573", "#C0392B"
PALE, INK = "#EEF3FA", "#1C2430"
NODE = {"blue": ("#DCE6F5", "#9DB6DC", "#16365C"),
        "green": ("#DCEBDC", "#9CC79B", INK),
        "purple": ("#E4DEF0", "#B7A9D6", INK),
        "yellow": ("#FBF1CE", "#E0C97A", INK),
        "neutral": ("#EEF1F5", "#CAD2DE", INK)}
OUT = Path(r"D:\gravity\ppt_jq\figs")

METHODS = ["SSV33.1 卫星重力模型", "最近邻插值（Verde）", "线性插值（Verde）",
           "普通克里金（PyKrige）", "频域融合", "卡尔曼滤波（本方法）"]
DATA = {
    "综合验证区": [4.674, 5.531, 5.056, 5.733, 4.575, 4.173],
    "南海-琉球": [5.252, 5.746, 5.504, 5.139, 5.152, 4.748],
    "菲律宾海中部": [1.607, 2.088, 2.280, 2.585, 1.563, 1.455],
}


def fig_results():
    """三个区域各一个横向条形图，本方法用 navy 实心、其余用浅蓝。"""
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.3), dpi=200,
                             gridspec_kw=dict(wspace=0.10))
    for ax, (region, vals) in zip(axes, DATA.items()):
        y = np.arange(len(METHODS))
        colors = [NAVY if i == len(METHODS) - 1 else LIGHT for i in range(len(METHODS))]
        ax.barh(y, vals, color=colors, height=0.62, zorder=3)
        for i, v in enumerate(vals):
            ax.text(v + 0.10, i, f"{v:.3f}", va="center", ha="left", fontsize=9.5,
                    color=NAVY if i == len(METHODS) - 1 else GREY,
                    fontweight="bold" if i == len(METHODS) - 1 else "normal")
        ax.set_yticks(y)
        ax.set_yticklabels(METHODS if ax is axes[0] else [""] * len(METHODS),
                           fontsize=10.5)
        for lbl in ax.get_yticklabels():
            if lbl.get_text().startswith("卡尔曼"):
                lbl.set_color(NAVY); lbl.set_fontweight("bold")
        ax.invert_yaxis()
        ax.set_title(region, fontsize=12, color=NAVY, fontweight="bold", pad=8)
        ax.set_xlim(0, max(vals) * 1.30)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.grid(axis="x", ls=":", lw=0.6, color="0.88", zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(LIGHT)
    axes[0].set_xlabel("RMSE (mGal)", fontsize=9.5)
    fig.savefig(OUT / "fig_results.png", bbox_inches="tight")
    plt.close(fig)
    print("fig_results.png")


def fig_roadmap():
    """下一步：三步路线图（水平流向）。"""
    fig, ax = plt.subplots(figsize=(11.6, 2.5), dpi=200)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 11.85); ax.set_ylim(0, 2.5); ax.axis("off")
    steps = [("①", "跨数据库检验", "改用公开发布的俯冲带剩余重力格网\n独立复验，排除训练与验证同源", "blue"),
             ("②", "不确定度应用", "逐像元方差已随场输出\n可用于标定可信区间", "green"),
             ("③", "补测规划", "优势依赖船测约束\n优先在缺船测海域补测", "yellow")]
    w, h, gap = 3.10, 1.62, 1.10          # 加大间距，给箭头两端留净空
    for k, (num, title, desc, kind) in enumerate(steps):
        x = 0.10 + k * (w + gap)
        fill, edge, txt = NODE[kind]
        ax.add_patch(mp.FancyBboxPatch((x, 0.42), w, h,
                                       boxstyle="round,pad=0.06",
                                       fc=fill, ec=edge, lw=1.5, zorder=3,
                                       clip_on=False))
        ax.text(x + 0.30, 0.42 + h - 0.34, num, fontsize=15, fontweight="bold",
                color=NAVY, va="center")
        ax.text(x + 0.78, 0.42 + h - 0.34, title, fontsize=13.5, fontweight="bold",
                color=NAVY, va="center")
        ax.text(x + 0.30, 0.42 + h - 0.92, desc, fontsize=10.5, color=INK,
                va="center", linespacing=1.5)
        if k < len(steps) - 1:
            ax.annotate("", xy=(x + w + gap - 0.30, 0.42 + h / 2),
                        xytext=(x + w + 0.18, 0.42 + h / 2),
                        arrowprops=dict(arrowstyle="-|>", lw=2.0, color=NAVY,
                                        shrinkA=0, shrinkB=0))
    fig.savefig(OUT / "fig_roadmap.png", bbox_inches="tight")
    plt.close(fig)
    print("fig_roadmap.png")


def fig_cover_map():
    """封面用研究区图：卫星重力背景 + 三个验证区框。"""
    import xarray as xr
    from scipy.ndimage import zoom
    ds = xr.open_dataset(r"D:\gravity\kalman_gravity\data\ssv32\grav_33.1.nc",
                         engine="h5netcdf")
    sub = ds["z"].sel(lon=slice(124, 153), lat=slice(12, 43)).values.astype(float)
    ds.close()
    sub = zoom(sub, 0.5, order=1)
    fig, ax = plt.subplots(figsize=(6.0, 4.6), dpi=200)
    ax.imshow(sub, extent=[124, 153, 12, 43], origin="lower", cmap="Blues_r",
              interpolation="nearest", aspect="auto")
    boxes = [("G", 132, 26, 20, 16), ("D", 124, 24, 8, 7), ("F", 139, 13, 13, 13)]
    for name, x0, y0, w, h in boxes:
        ax.add_patch(mp.Rectangle((x0, y0), w, h, fill=False, ec=RED, lw=1.8, zorder=5))
        ax.text(x0 + 0.45, y0 + h - 1.5, name, color=RED, fontsize=12,
                fontweight="bold", zorder=6)
    ax.set_xlim(124, 153); ax.set_ylim(12, 43)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#FFFFFF"); s.set_linewidth(2)
    fig.savefig(OUT / "fig_cover_map.png", bbox_inches="tight",
                facecolor=NAVY_D if False else "white")
    plt.close(fig)
    print("fig_cover_map.png")


if __name__ == "__main__":
    fig_results()
    fig_roadmap()
    fig_cover_map()
