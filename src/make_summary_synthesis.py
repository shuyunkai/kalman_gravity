"""总结页：5 步流程 + 为什么这样做。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = r"output_real/three_parts/newdeck"
NAVY, GOLD, RED, PALE = "#0B2545", "#C9A227", "#C0392B", "#EEF3F9"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})

fig, ax = plt.subplots(figsize=(13.6, 5.2), dpi=220)
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

ax.text(0.02, 0.955, "做了什么", fontsize=11, color=GOLD, fontweight="bold")
steps = [
    ("输入", "1′ + 船测 + 地形"),
    ("初始先验", "SSV 块平均 + 地形高频"),
    ("相干预测", "$A=(1-wC^2)G_\\sigma+wC^2I$"),
    ("观测更新", "块观测 + 船测邻域，6 轮 + RTS"),
    ("输出", "15″ 场 + $P$"),
]
for i, (tag, txt) in enumerate(steps):
    y = 0.78 - i*0.155
    ax.add_patch(FancyBboxPatch((0.02, y), 0.60, 0.115,
                                boxstyle="round,pad=0.010,rounding_size=0.02",
                                fc="#F7F9FC", ec=NAVY, lw=1))
    ax.text(0.045, y+0.057, f"{i+1}", fontsize=8, color=RED,
            fontweight="bold", va="center")
    ax.text(0.075, y+0.078, tag, fontsize=8, color=GOLD, va="center")
    ax.text(0.075, y+0.036, txt, fontsize=8.4, color=NAVY, va="center")
    if i < 4:
        ax.add_patch(FancyArrowPatch((0.32, y-0.005), (0.32, y-0.033),
                                     arrowstyle="-|>", mutation_scale=11,
                                     color="#78909C", lw=1.1))

ax.add_patch(FancyBboxPatch((0.66, 0.12), 0.32, 0.80,
                            boxstyle="round,pad=0.012,rounding_size=0.03",
                            fc="#FFF8E7", ec=GOLD, lw=1))
ax.text(0.82, 0.86, "为什么这样做", ha="center", fontsize=11,
        color=NAVY, fontweight="bold")
why = [
    "1′ 产品 20 km 以下被平滑，\n需要补 5–20 km 短波。",
    "C 由地形结构张量得到，\n脊线少平滑、平地正常平滑。",
    "块观测、船测和地形先验\n放进同一递推，避免各做一套场。",
    "先验只作初始场，后续由\n观测逐轮修正，避免重复加权。",
]
for i, s in enumerate(why):
    y = 0.72 - i*0.155
    ax.text(0.69, y, "•", fontsize=11, color=RED, va="top")
    ax.text(0.715, y, s, fontsize=8.1, color="#1A1A1A", va="top")

fig.savefig(f"{ROOT}/fig_new_summary_synthesis.png", bbox_inches="tight")
print("saved")
