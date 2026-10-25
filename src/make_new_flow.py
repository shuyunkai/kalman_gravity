"""两行流程图：数据→KF→15″ / 15″→带通→THG→线性体→GM9。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 10})
fig, ax = plt.subplots(figsize=(13.6, 5.0), dpi=220)
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)


def box(x, y, w, h, t, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc="#EEF3F9", ec="#0B2545", lw=1.1))
    ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=fs,
            color="#0B2545")


def arr(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#5B6573", lw=1.4))


for lab, y in (("卫星 1′", 0.78), ("船测重力", 0.60), ("地形 GEBCO", 0.42)):
    box(0.02, y, 0.12, 0.14, lab, fs=9)
box(0.18, 0.48, 0.15, 0.36, "相干保持 KF\n块观测 + 相干预测\nRTS 平滑", fs=9)
box(0.39, 0.56, 0.12, 0.22, "15″ 重力场\n1.85 km", fs=9)
box(0.39, 0.10, 0.15, 0.30, "5–20 km 带通\nTHG 脊线\n细化/过滤", fs=9)
box(0.58, 0.10, 0.12, 0.30, "线性体\n(单像元)", fs=9)
box(0.74, 0.10, 0.11, 0.30, "线性体\n(滤波后)", fs=9)
box(0.88, 0.10, 0.10, 0.30, "GM9\n断层", fs=9)

for y in (0.85, 0.67, 0.49):
    arr(0.14, y, 0.18, 0.66)
arr(0.33, 0.66, 0.39, 0.66)
arr(0.45, 0.56, 0.45, 0.40)
arr(0.54, 0.25, 0.58, 0.25)
arr(0.70, 0.25, 0.74, 0.25)
arr(0.85, 0.25, 0.88, 0.25)
ax.text(0.5, 0.95, "15″ 重力场重建与断裂识别", ha="center", fontsize=13,
        color="#0B2545", fontweight="bold")
fig.tight_layout()
fig.savefig(r"output_real/three_parts/newdeck/fig_new_flow_fault.png",
            bbox_inches="tight")
print("ok")
