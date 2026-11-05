"""结论页：方法—场—应用三栏。"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, GOLD, RED, PALE = "#0B2545", "#C9A227", "#C0392B", "#EEF3F9"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 8})

fig, ax = plt.subplots(figsize=(13.4, 4.2), dpi=240)
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
cols = [
    ("方法优势", "相干保持 KF", ["结构张量 C 控制平滑", "脊线与短波不被抹平", "空间域解算，无 FFT 振铃", "6 轮 + RTS 修正量收敛"]),
    ("重力场优势", "15″ 重建结果", ["15″ 格网，补 5–20 km", "长波背景保持", "块边界无台阶", "逐像元方差可评估"]),
    ("应用优势", "直接可用", ["重力匹配导航", "海底地形 / 工程", "资源浅层", "断裂识别"]),
]
w, gap = 0.29, 0.055
x0 = 0.03
for i, (head, sub, items) in enumerate(cols):
    x = x0 + i*(w+gap)
    ax.add_patch(FancyBboxPatch((x, 0.14), w, 0.70,
                                boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc="white", ec="#C9D3DF", lw=1.0))
    ax.add_patch(FancyBboxPatch((x, 0.70), w, 0.14,
                                boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=NAVY, ec=NAVY))
    ax.text(x+w/2, 0.795, head, color="white", fontsize=10.5,
            fontweight="bold", ha="center", va="center")
    ax.text(x+w/2, 0.735, sub, color=GOLD, fontsize=7.5,
            ha="center", va="center")
    for j, t in enumerate(items):
        ax.text(x+0.03, 0.60-j*0.115, "•", color=RED, fontsize=10, va="center")
        ax.text(x+0.055, 0.60-j*0.115, t, color="#1A1A1A", fontsize=7.8, va="center")
    if i < 2:
        ax.add_patch(FancyArrowPatch((x+w+0.006, 0.49), (x+w+gap-0.006, 0.49),
                                     arrowstyle="-|>", mutation_scale=14,
                                     color=GOLD, lw=1.6))
fig.savefig(OUT / "fig_conclusion_value.png", bbox_inches="tight")
print("saved")
