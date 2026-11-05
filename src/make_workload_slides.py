"""工作量页：方法链条与三区消融。"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, GOLD, RED, PALE = "#0B2545", "#C9A227", "#C0392B", "#EEF3F9"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 8})

# ---- method chain ----
fig, ax = plt.subplots(figsize=(13.4, 4.6), dpi=220)
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
steps = [
    ("01", "多源统一", ["4 套 1′ 产品", "船测 + 地形", "统一到 15″ 状态"]),
    ("02", "块观测 H", ["1′ 块平均", "16×16 个 15″ 格点", "R/Q 定权"]),
    ("03", "结构张量 C", ["梯度 → 张量平滑", "特征分解", "相干归一化"]),
    ("04", "自适应预测 A", ["A=(1−wC²)Gσ+wC²I", "脊线少平滑", "平地正常平滑"]),
    ("05", "迭代与验证", ["6 轮 + RTS", "三区留出航次", "消融/断层应用"]),
]
w, gap = 0.172, 0.028
for i, (num, head, items) in enumerate(steps):
    x = 0.02 + i*(w+gap)
    ax.add_patch(FancyBboxPatch((x, 0.28), w, 0.52,
                                boxstyle="round,pad=0.006,rounding_size=0.018",
                                fc="white", ec="#C9D3DF", lw=0.9))
    ax.add_patch(FancyBboxPatch((x, 0.70), w, 0.10,
                                boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=NAVY, ec=NAVY))
    ax.text(x+0.018, 0.75, num, color=GOLD, fontsize=8, fontweight="bold", va="center")
    ax.text(x+w/2, 0.75, head, color="white", fontsize=8.5,
            fontweight="bold", ha="center", va="center")
    for j, t in enumerate(items):
        ax.text(x+0.014, 0.61-j*0.105, "•", color=RED, fontsize=9, va="center")
        ax.text(x+0.035, 0.61-j*0.105, t, color="#1A1A1A", fontsize=7.2, va="center")
    if i < 4:
        ax.add_patch(FancyArrowPatch((x+w+0.004, 0.54), (x+w+gap-0.004, 0.54),
                                     arrowstyle="-|>", mutation_scale=10,
                                     color=GOLD, lw=1.2))
ax.text(0.5, 0.13, "工作量落在：数据统一、块观测算子、C 的张量估计、w 的自适应、6 轮解算与三区验证",
        ha="center", fontsize=8, color=NAVY)
fig.savefig(OUT / "fig_workload_chain.png", bbox_inches="tight")
plt.close(fig)

# ---- ablation ----
regions = ["南海-琉球", "九州东侧", "日本海沟"]
rows = [("双线性", [5.344, 5.514, 3.960]),
        ("标准 KF", [4.951, 5.510, 3.883]),
        ("相干保持 KF", [4.7205, 4.1491, 3.4157])]
fig = plt.figure(figsize=(13.4, 4.7), dpi=220)
ax = fig.add_axes([0.04, 0.30, 0.56, 0.55]); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
cols = [0.0, 0.34, 0.56, 0.78]; widths = [0.34, 0.22, 0.22, 0.22]
head = ["方法 / 区域", "南海-琉球", "九州东侧", "日本海沟"]
for c, w, h in zip(cols, widths, head):
    ax.add_patch(plt.Rectangle((c, 0.78), w, 0.18, fc=NAVY, ec="white", lw=0.8))
    ax.text(c+w/2, 0.87, h, color="white", fontsize=7.6, ha="center", va="center")
for i, (name, vals) in enumerate(rows):
    y = 0.78 - (i+1)*0.19
    for c, w, val in zip(cols, widths, [name]+vals):
        ax.add_patch(plt.Rectangle((c, y), w, 0.19,
                                   fc=PALE if i % 2 == 0 else "white",
                                   ec="#D5DCE5", lw=0.5))
        ax.text(c+w/2, y+0.095, f"{val:.2f}" if isinstance(val, float) else val,
                fontsize=7.4 if isinstance(val, float) else 7.0,
                ha="center", va="center",
                color=RED if i == 2 else "#1A1A1A",
                fontweight="bold" if i == 2 else "normal")
ax.text(0.0, 0.12, "标准 KF 已含 1′ 块观测与地形先验；相干保持 KF 再加入 C 与自适应预测。",
        fontsize=6.6, color="#44546A")
# improvement bars
ax2 = fig.add_axes([0.66, 0.18, 0.30, 0.62])
x = np.arange(3); width = 0.34
std = [(5.344-4.951)/5.344*100, (5.514-5.510)/5.514*100, (3.960-3.883)/3.960*100]
ckf = [(5.344-4.7205)/5.344*100, (5.514-4.1491)/5.514*100, (3.960-3.4157)/3.960*100]
ax2.bar(x-width/2, std, width, color="#9AA7B4", label="标准 KF")
ax2.bar(x+width/2, ckf, width, color=RED, label="相干保持 KF")
ax2.set_xticks(x); ax2.set_xticklabels(regions, fontsize=6.2)
ax2.set_ylabel("相对双线性的 RMSE 改善 (%)", fontsize=6.5)
ax2.tick_params(labelsize=6)
ax2.legend(fontsize=6, frameon=False, loc="upper left")
ax2.grid(axis="y", alpha=0.2)
ax2.set_title("改善幅度", fontsize=7, pad=2)
fig.savefig(OUT / "fig_ablation.png", bbox_inches="tight")
print("saved")
