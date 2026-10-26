"""贡献页：公式层面的标准 KF vs 相干保持 KF。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = r"output_real/three_parts/newdeck"
NAVY, RED, GOLD, PALE = "#0B2545", "#C0392B", "#C9A227", "#D6E2F0"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 10})

fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.0), dpi=220,
                         gridspec_kw={"width_ratios": [1, 1, 1.15], "wspace": 0.28})

lam = np.logspace(0, 2, 400)  # km
sig = 4.0
H = np.exp(-0.5 * (2*np.pi*sig/lam)**2)

ax = axes[0]
ax.semilogx(lam, H, color=NAVY, lw=2)
ax.axvspan(5, 20, color="#F2B8B5", alpha=0.25)
ax.set_xlabel("波长 (km)", fontsize=9); ax.set_ylabel("$H(k)$", fontsize=9)
ax.set_title("(a) 标准 KF：$A=G_\\sigma$", fontsize=11)
ax.text(0.5, -0.22, "$H(k)=e^{-k^2\\sigma^2/2}$", transform=ax.transAxes,
        ha="center", fontsize=11, color=RED)
ax.grid(alpha=0.2, which="both"); ax.tick_params(labelsize=8)
ax.annotate("5–20 km 短波被压低", xy=(12, 0.15), xytext=(25, 0.55),
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2),
            fontsize=8, color=RED)

ax = axes[1]
for c, col, ls in ((0.0, "#90A4AE", ":"), (0.5, "#2E5C8A", "--"), (1.0, NAVY, "-")):
    Hi = 1 - c**2 * (1 - H)
    ax.semilogx(lam, Hi, color=col, ls=ls, lw=2, label=f"$c={c}$")
ax.axvspan(5, 20, color="#F2B8B5", alpha=0.25)
ax.set_xlabel("波长 (km)", fontsize=9); ax.set_ylabel("$H_i(k)$", fontsize=9)
ax.set_title("(b) 相干保持：$A=(1-wC^2)G_\\sigma+wC^2$", fontsize=10)
ax.legend(fontsize=8, frameon=False, loc="lower right")
ax.grid(alpha=0.2, which="both"); ax.tick_params(labelsize=8)
ax.text(0.5, -0.22, "$H_i(k)=1-wc_i^2(1-e^{-k^2\\sigma^2/2})$",
        transform=ax.transAxes, ha="center", fontsize=10, color=RED)

ax = axes[2]; ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.text(0.5, 0.94, "(c) 改在哪、解决什么", ha="center", fontsize=12,
        color=NAVY, fontweight="bold")
rows = [
    ("原式", "$x^-=G_\\sigma x$"),
    ("改后", "$x^-=G_\\sigma x+wC^2(x-G_\\sigma x)$"),
    ("$c=1$", "$A\\approx I$：脊线不平滑"),
    ("$c=0$", "$A\\approx G_\\sigma$：平坦区照常平滑"),
    ("5–20 km", "$H_i$ 由 $e^{-k^2\\sigma^2/2}$ 提高到 1"),
]
for i, (a, b) in enumerate(rows):
    y = 0.78 - i*0.16
    ax.text(0.03, y, a, fontsize=11, color=RED, fontweight="bold", va="center")
    ax.text(0.22, y, b, fontsize=10.5, color="#1A1A1A", va="center")
fig.savefig(f"{ROOT}/fig_new_contribution_formula.png", bbox_inches="tight")
print("saved")
