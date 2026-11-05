"""总体精度页：三区四种方法 RMSE。"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
NAVY, RED, PALE = "#0B2545", "#C0392B", "#EEF3F9"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 8})
regions = ["南海-琉球", "九州东侧", "日本海沟"]
rows = [("双线性", [5.34, 5.51, 3.96]),
        ("频域融合", [5.34, 5.17, 3.91]),
        ("LSC", [5.38, 4.95, 4.21]),
        ("相干保持 KF", [4.72, 4.15, 3.42])]
fig = plt.figure(figsize=(13.4, 4.8), dpi=220)
ax = fig.add_axes([0.05, 0.24, 0.58, 0.62]); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
cols = [0.0, 0.34, 0.56, 0.78]; widths = [0.34, 0.22, 0.22, 0.22]
for c, w, h in zip(cols, widths, ["方法 / 区域"]+regions):
    ax.add_patch(plt.Rectangle((c, 0.80), w, 0.18, fc=NAVY, ec="white", lw=0.8))
    ax.text(c+w/2, 0.89, h, color="white", fontsize=7.8, ha="center", va="center")
for i, (name, vals) in enumerate(rows):
    y = 0.80 - (i+1)*0.19
    for c, w, val in zip(cols, widths, [name]+vals):
        ax.add_patch(plt.Rectangle((c, y), w, 0.19,
                                   fc=PALE if i % 2 == 0 else "white",
                                   ec="#D5DCE5", lw=0.5))
        ax.text(c+w/2, y+0.095, f"{val:.2f}" if isinstance(val, float) else val,
                fontsize=7.6 if isinstance(val, float) else 7.2,
                ha="center", va="center",
                color=RED if i == 3 else "#1A1A1A",
                fontweight="bold" if i == 3 else "normal")
# bars
ax2 = fig.add_axes([0.68, 0.18, 0.30, 0.64])
methods = ["双线性", "频域", "LSC", "相干保持 KF"]
series = {
    "南海—琉球": [5.34, 5.34, 5.38, 4.72],
    "九州东侧": [5.51, 5.17, 4.95, 4.15],
    "日本海沟": [3.96, 3.91, 4.21, 3.42],
}
colors = ["#C0392B", "#2E86AB", "#7F8C8A"]
for (name, vals), c in zip(series.items(), colors):
    ax2.plot(range(4), vals, "o-", color=c, lw=1.4, ms=3.5, label=name)
ax2.set_xticks(range(4)); ax2.set_xticklabels(methods, fontsize=5.6, rotation=18)
ax2.set_ylabel("RMSE (mGal)", fontsize=6.5); ax2.tick_params(labelsize=5.8)
ax2.set_ylim(3.2, 5.7); ax2.grid(axis="y", alpha=0.2)
ax2.legend(fontsize=5.2, frameon=False, loc="upper right")
ax2.set_title("三区四类方法 RMSE", fontsize=7.0, pad=2)
for i, v in enumerate(series["日本海沟"]):
    ax2.annotate(f"{v:.2f}", (i, v), textcoords="offset points", xytext=(0, -8),
                 ha="center", fontsize=5.0, color="#333333")
fig.text(0.5, 0.02, "RMSE 3.42–4.72 mGal；三区均优于三种常用方法",
         ha="center", fontsize=8, color=RED)
fig.savefig(OUT / "fig_overall_accuracy.png", bbox_inches="tight")
print("saved")
