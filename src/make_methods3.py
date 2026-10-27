"""第 4 页：三种方法解算 15″ 的流程（三条泳道）。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = r"output_real/three_parts/newdeck"
NAVY, GOLD, RED, LIGHT = "#0B2545", "#C9A227", "#C0392B", "#EEF3F9"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 9})

fig, ax = plt.subplots(figsize=(13.6, 4.7), dpi=220)
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

lanes = [
    ("插值", ["1′ 格网 + 船测", "核函数\n双线性 / 克里金", "逐点解算\n15″ 格点", "15″ 场\n短波未增加"],
     "$\\hat z(x_0)=\\sum w_i z_i$", "无新增短波"),
    ("频域融合", ["测高重力 + 船测", "谱权 $W(k)$\n长波 / 短波", "逆变换到\n15″ 格点", "15″ 场\n短波来自船测/先验"],
     "$Z=WZ_{alt}+(1-W)Z_{ship}$", "短波来自船测"),
    ("最小二乘配置", ["观测 + 协方差", "协方差函数\n$C(r)$", "配置预测\n15″ 格点", "15″ 场\n短波来自协方差"],
     "$\\hat x=C_{xs}(C_{ss}+C_{nn})^{-1}z$", "带误差输出"),
]

for k, (name, steps, eq, note) in enumerate(lanes):
    x = 0.02 + k * 0.325
    w = 0.29
    ax.add_patch(FancyBboxPatch((x, 0.04), w, 0.90,
                                boxstyle="round,pad=0.012,rounding_size=0.03",
                                fc="#F7F9FC", ec="#B0BEC5", lw=1))
    ax.add_patch(FancyBboxPatch((x+0.02, 0.82), w-0.04, 0.09,
                                boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc=NAVY, ec=NAVY))
    ax.text(x+w/2, 0.865, name, ha="center", va="center", fontsize=13,
            color=GOLD, fontweight="bold")
    for i, txt in enumerate(steps):
        y = 0.63 - i*0.15
        fc = "#FFF8E7" if i == 3 else "white"
        ax.add_patch(FancyBboxPatch((x+0.035, y), w-0.07, 0.11,
                                    boxstyle="round,pad=0.01,rounding_size=0.02",
                                    fc=fc, ec=NAVY, lw=1))
        ax.text(x+w/2, y+0.055, txt, ha="center", va="center", fontsize=9)
        ax.text(x+0.012, y+0.055, str(i+1), ha="center", va="center",
                fontsize=8, color=RED, fontweight="bold")
        if i < 3:
            ax.add_patch(FancyArrowPatch((x+w/2, y-0.005), (x+w/2, y-0.045),
                                         arrowstyle="-|>", mutation_scale=12,
                                         color="#78909C", lw=1.1))
    ax.text(x+w/2, 0.075, eq, ha="center", fontsize=8.5, color=RED)

ax.text(0.5, 0.985, "三种方法都把 1′ 与船测解算到 15″ 格点，插值不新增短波",
        ha="center", fontsize=11, color=NAVY, fontweight="bold")
fig.savefig(f"{ROOT}/fig_new_methods_3lanes.png", bbox_inches="tight")
print("saved")
