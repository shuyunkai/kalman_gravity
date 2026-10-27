"""贡献页：公式层面的标准 KF vs 相干保持 KF。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = r"output_real/three_parts/newdeck"
NAVY, RED, GOLD, PALE = "#0B2545", "#C0392B", "#C9A227", "#D6E2F0"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 10})

fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.0), dpi=220,
                         gridspec_kw={"width_ratios": [1, 1, 1.15], "wspace": 0.28})

lam = np.logspace(0, 2, 400)  # km
sig = 4.0
T = np.exp(-0.5 * (2*np.pi*sig/lam)**2)

ax = axes[0]
ax.semilogx(lam, T, color=NAVY, lw=2)
ax.axvspan(5, 20, color="#F2B8B5", alpha=0.25)
ax.set_xlabel("波长 (km)", fontsize=9); ax.set_ylabel("$T(k)$", fontsize=9)
ax.set_title("(a) 标准 KF：$A=G_\\sigma$", fontsize=11)
ax.text(0.5, -0.22, "$T(k)=e^{-k^2\\sigma^2/2}$", transform=ax.transAxes,
        ha="center", fontsize=11, color=RED)
ax.grid(alpha=0.2, which="both"); ax.tick_params(labelsize=8)
ax.annotate("5–20 km 短波被压低", xy=(12, 0.15), xytext=(25, 0.55),
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2),
            fontsize=8, color=RED)

ax = axes[1]
for c, col, ls in ((0.0, "#90A4AE", ":"), (0.5, "#2E5C8A", "--"), (1.0, NAVY, "-")):
    Ti = 1 - c**2 * (1 - T)
    ax.semilogx(lam, Ti, color=col, ls=ls, lw=2, label=f"$c_i={c}$")
ax.axvspan(5, 20, color="#F2B8B5", alpha=0.25)
ax.set_xlabel("波长 (km)", fontsize=9); ax.set_ylabel("$T_i(k)$", fontsize=9)
ax.set_title("(b) 相干保持：$A=(1-wC^2)G_\\sigma+wC^2I$", fontsize=9.5)
ax.legend(fontsize=8, frameon=False, loc="lower right")
ax.grid(alpha=0.2, which="both"); ax.tick_params(labelsize=8)
ax.text(0.5, -0.22, "$T_i(k)=1-wc_i^2(1-e^{-k^2\\sigma^2/2})$",
        transform=ax.transAxes, ha="center", fontsize=10, color=RED)

ax = axes[2]; ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.text(0.5, 0.97, "(c) $C$ 的取值", ha="center", fontsize=12,
        color=NAVY, fontweight="bold")
ax.text(0.03, 0.88, "1. 由 GEBCO 地形得布格高频 $h_p$", fontsize=8.5, va="center")
ax.text(0.03, 0.79, "2. 求梯度", fontsize=8.5, va="center")
ax.text(0.5, 0.70, "$g_x=\\partial h/\\partial x,\\quad g_y=\\partial h/\\partial y$",
        fontsize=9, ha="center", va="center")
ax.text(0.03, 0.60, "3. 构造结构张量", fontsize=8.5, va="center")
ax.text(0.5, 0.52, "$J_{xx}=G_8(g_x^2),\\quad J_{yy}=G_8(g_y^2)$",
        fontsize=8, ha="center", va="center")
ax.text(0.5, 0.45, "$J_{xy}=G_8(g_xg_y)$", fontsize=8, ha="center", va="center")
ax.text(0.03, 0.37, "$G_8$：8 格点高斯平滑，15″ 下约 3.7 km",
        fontsize=7.5, va="center", color="#44546A")
ax.text(0.03, 0.30, "4. 相干", fontsize=8.5, va="center")
ax.text(0.5, 0.19,
        "$C_i=\\frac{\\sqrt{(J_{xx}-J_{yy})^2+4J_{xy}^2}}{J_{xx}+J_{yy}+\\epsilon}$",
        fontsize=9.5, ha="center", va="center")
ax.text(0.03, 0.10, "5. 截断并平方：$0\\leq C_i\\leq1$", fontsize=8.5, va="center")
ax.text(0.03, 0.04, "权重 $wC_i^2$", fontsize=8.5, va="center")
fig.savefig(f"{ROOT}/fig_new_contribution_formula.png", bbox_inches="tight")
print("saved")
