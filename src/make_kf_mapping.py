"""第 7 页：标准 KF 环节与三类方法缺点的对应关系。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = r"output_real/three_parts/newdeck"
NAVY, RED, GOLD, LIGHT = "#0B2545", "#C0392B", "#C9A227", "#EEF3F9"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})

fig, ax = plt.subplots(figsize=(13.4, 5.2), dpi=220)
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

ax.text(0.5, 0.96, "标准 KF 的各个环节，分别对应前面三类方法的缺点", ha="center",
        fontsize=12, color=NAVY, fontweight="bold")
ax.text(0.16, 0.87, "前面方法的缺点", ha="center", fontsize=10, color=RED)
ax.text(0.49, 0.87, "KF 对应环节", ha="center", fontsize=10, color=NAVY)
ax.text(0.82, 0.87, "怎么解决", ha="center", fontsize=10, color=GOLD)

rows = [
    ("插值：短波被平滑", "② 观测更新", "块观测 + 船测点\n把短波重新注入"),
    ("频域：边界振铃", "空间域递推", "无 FFT 周期边界\n局部更新"),
    ("LSC：协方差选择/求逆", "③ 增益 + R/Q", "块增益\n避免全局协方差求逆"),
    ("共同：噪声/相关未显式", "P / R / Q + 相干预测", "噪声、方差、空间相关\n进入增益"),
]
for i, (left, mid, right) in enumerate(rows):
    y = 0.66 - i*0.18
    ax.add_patch(FancyBboxPatch((0.02, y), 0.28, 0.13,
                                boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc="#FCEDEA", ec=RED, lw=1))
    ax.text(0.16, y+0.065, left, ha="center", va="center", fontsize=8.5)
    ax.add_patch(FancyBboxPatch((0.37, y), 0.24, 0.13,
                                boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc=LIGHT, ec=NAVY, lw=1))
    ax.text(0.49, y+0.065, mid, ha="center", va="center", fontsize=9, color=NAVY)
    ax.add_patch(FancyBboxPatch((0.68, y), 0.30, 0.13,
                                boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc="#FFF8E7", ec=GOLD, lw=1))
    ax.text(0.83, y+0.065, right, ha="center", va="center", fontsize=8.5)
    ax.add_patch(FancyArrowPatch((0.30, y+0.065), (0.37, y+0.065),
                                 arrowstyle="-|>", mutation_scale=13,
                                 color="#78909C", lw=1.2))
    ax.add_patch(FancyArrowPatch((0.61, y+0.065), (0.68, y+0.065),
                                 arrowstyle="-|>", mutation_scale=13,
                                 color="#78909C", lw=1.2))

ax.text(0.5, 0.03, "先由 KF 统一表达观测、噪声和协方差，再由后续相干预测与局部化改进",
        ha="center", fontsize=9, color=RED)
fig.savefig(f"{ROOT}/fig_new_kf_mapping.png", bbox_inches="tight")
print("saved")
