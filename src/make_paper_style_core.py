"""论文风格核心图：1′局限、应用对比、算法对比。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize, LightSource
import sys
sys.path.insert(0, str(Path(__file__).parent))
from scale_decomposition import bandpass
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei"], "font.family": "sans-serif",
    "font.size": 6.5, "axes.linewidth": 0.5, "axes.labelsize": 6.5,
    "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 2, "ytick.major.size": 2,
    "axes.unicode_minus": False,
})

d = np.load(OUT / "validation_G_fields.npz")
lon, lat = d["lon"], d["lat"]
dx = (lon[1]-lon[0])*111320*np.cos(np.radians(lat.mean()))
dy = (lat[1]-lat[0])*111320
bathy = np.load(ROOT / "output_real/three_parts/ship_regions/G_Japan.npz")["bathy"]


def axes_style(ax, ext, xlab=False, ylab=False):
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([ext[0], (ext[0]+ext[1])/2, ext[1]])
    ax.set_yticks([ext[2], (ext[2]+ext[3])/2, ext[3]])
    ax.tick_params(pad=1)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    if xlab:
        ax.set_xlabel("经度 (°E)", labelpad=1)
    if ylab:
        ax.set_ylabel("纬度 (°N)", labelpad=1)


def terrain(ax, ext, xlab=False, ylab=False):
    sx = (lon >= ext[0]) & (lon <= ext[1]); sy = (lat >= ext[2]) & (lat <= ext[3])
    z = bathy[np.ix_(sy, sx)]
    ls = LightSource(azdeg=315, altdeg=45)
    ax.imshow(ls.shade(z, cmap=plt.cm.Greys, vert_exag=0.5, blend_mode="overlay"),
              extent=ext, origin="lower", aspect="auto")
    axes_style(ax, ext, xlab, ylab)


def gravity(ax, z, ext, v=None, cmap="RdBu_r", xlab=False, ylab=False):
    if v is None:
        v = np.nanpercentile(np.abs(z), 98)
    im = ax.imshow(z, cmap=cmap, extent=ext, origin="lower", aspect="auto",
                   vmin=-v, vmax=v)
    axes_style(ax, ext, xlab, ylab)
    return im


def horizontal_cbar(fig, rect, cmap, label, ticks=True):
    cax = fig.add_axes(rect)
    ColorbarBase(cax, cmap=plt.get_cmap(cmap), norm=Normalize(0, 1),
                 orientation="horizontal")
    cax.set_xticks([]); cax.set_yticks([])
    cax.set_xlabel(label, fontsize=6, labelpad=1)
    for s in cax.spines.values():
        s.set_linewidth(0.4)


# ---------- 1′ limitation: terrain + 1′/15″ 5-20 ----------
ext = [138, 142, 33.5, 36]
sx = (lon >= ext[0]) & (lon <= ext[1]); sy = (lat >= ext[2]) & (lat <= ext[3])
g1 = bandpass(gaussian_filter(d["ssv"], 8e3/dx), dx, dy, 5e3, 20e3)[np.ix_(sy, sx)]
g2 = bandpass(d["ckf"], dx, dy, 5e3, 20e3)[np.ix_(sy, sx)]
fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.4), dpi=260,
                         gridspec_kw={"wspace": 0.08})
terrain(axes[0], ext, xlab=True, ylab=True)
im = gravity(axes[1], g1, ext, xlab=True)
gravity(axes[2], g2, ext, xlab=True)
for ax, lab in zip(axes, ["(a)", "(b)", "(c)"]):
    ax.text(0.03, 0.82, lab, transform=ax.transAxes, ha="left", va="top",
            fontsize=7, color="k")
cax = fig.add_axes([0.35, 0.03, 0.55, 0.025])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("重力异常 (mGal)", fontsize=6); cb.ax.tick_params(labelsize=5)
fig.savefig(OUT / "paper_limitation3.png", bbox_inches="tight")

# ---------- application comparison ----------
cols = [("导航：5–20 km 梯度", "grad"),
        ("地形/工程：5–20 km", "band"),
        ("资源/浅层：5–20 km 垂向梯度", "vert")]
g1f = bandpass(gaussian_filter(d["ssv"], 8e3/dx), dx, dy, 5e3, 20e3)
g2f = bandpass(d["ckf"], dx, dy, 5e3, 20e3)
fields = []
for _, key in cols:
    if key == "grad":
        a = np.hypot(*np.gradient(g1f)); b = np.hypot(*np.gradient(g2f))
    elif key == "vert":
        a = np.gradient(g1f, axis=0); b = np.gradient(g2f, axis=0)
    else:
        a, b = g1f, g2f
    fields.append((a[np.ix_(sy, sx)], b[np.ix_(sy, sx)]))
fig = plt.figure(figsize=(9.4, 4.2), dpi=260)
lefts = [0.04, 0.355, 0.67]; w = 0.29; hs = 0.33
for j, ((title, key), (a, b)) in enumerate(zip(cols, fields)):
    for i, z in enumerate([a, b]):
        v = np.nanpercentile(np.abs(z), 98)
        ax = fig.add_axes([lefts[j], 0.52-i*0.40, w, hs])
        cmap = "Reds" if key == "grad" else ("RdBu_r" if key == "band" else "PRGn")
        im = ax.imshow(z, cmap=cmap, extent=ext, origin="lower", aspect="auto",
                       vmin=(0 if key == "grad" else -v),
                       vmax=(v if key == "grad" else v))
        axes_style(ax, ext, xlab=(i == 1), ylab=(j == 0))
        ax.text(0.03, 0.82, f"({chr(97+j+3*i)})", transform=ax.transAxes,
                ha="left", va="top", fontsize=7)
        if i == 0:
            ax.set_title(title, fontsize=6.5, pad=2)
horizontal_cbar(fig, [0.08, 0.02, 0.18, 0.02], "Reds", "梯度 (mGal/km)")
horizontal_cbar(fig, [0.41, 0.02, 0.18, 0.02], "RdBu_r", "重力异常 (mGal)")
horizontal_cbar(fig, [0.72, 0.02, 0.18, 0.02], "PRGn", "垂向梯度 (mGal/km)")
fig.savefig(OUT / "paper_apps.png", bbox_inches="tight")

# ---------- algorithm comparison local ----------
ext2 = [139.2, 140.4, 34.5, 35.6]
sx2 = (lon >= ext2[0]) & (lon <= ext2[1]); sy2 = (lat >= ext2[2]) & (lat <= ext2[3])
flds = [("双线性", bandpass(d["ssv"], dx, dy, 5e3, 20e3)),
        ("频域融合", bandpass(d["freq"], dx, dy, 5e3, 20e3)),
        ("LSC", bandpass(d["lsc"], dx, dy, 5e3, 20e3)),
        ("相干保持 KF", bandpass(d["ckf"], dx, dy, 5e3, 20e3))]
fig = plt.figure(figsize=(9.4, 4.0), dpi=260)
terrain(fig.add_axes([0.04, 0.54, 0.29, 0.36]), ext2, ylab=True)
terrain_ax = fig.axes[-1]
terrain_ax.text(0.03, 0.82, "(a)", transform=terrain_ax.transAxes, va="top", fontsize=7)
pos = [[0.355, 0.54], [0.67, 0.54], [0.04, 0.12], [0.355, 0.12]]
v = np.nanpercentile(np.abs(np.concatenate(
    [f[np.ix_(sy2, sx2)].ravel() for _, f in flds])), 98)
for k, ((name, f), p) in enumerate(zip(flds, pos)):
    ax = fig.add_axes([p[0], p[1], 0.29, 0.36])
    im = gravity(ax, f[np.ix_(sy2, sx2)], ext2, v=v, xlab=(k >= 2), ylab=(k == 2))
    ax.text(0.03, 0.82, f"({chr(98+k)})", transform=ax.transAxes, va="top", fontsize=7)
    ax.set_title(name + " 5–20 km", fontsize=6.5, pad=2)
ax = fig.add_axes([0.67, 0.12, 0.29, 0.36])
dl = (flds[3][1] - flds[2][1])[np.ix_(sy2, sx2)]
vd = np.nanpercentile(np.abs(dl), 98)
imd = ax.imshow(dl, cmap="PuOr_r", extent=ext2, origin="lower", aspect="auto",
                vmin=-vd, vmax=vd)
axes_style(ax, ext2, xlab=True)
ax.text(0.03, 0.82, "(f)", transform=ax.transAxes, va="top", fontsize=7)
ax.set_title("相干保持 KF − LSC", fontsize=6.5, pad=2)
cax = fig.add_axes([0.40, 0.02, 0.22, 0.02])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("重力异常 (mGal)", fontsize=6); cb.ax.tick_params(labelsize=5)
cax2 = fig.add_axes([0.72, 0.02, 0.22, 0.02])
cb2 = fig.colorbar(imd, cax=cax2, orientation="horizontal")
cb2.set_label("差值 (mGal)", fontsize=6); cb2.ax.tick_params(labelsize=5)
fig.savefig(OUT / "paper_algorithms.png", bbox_inches="tight")
print("saved")
