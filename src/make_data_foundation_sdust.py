"""数据基础页：SDUST 先验、SSV 块观测、船测、GEBCO、1′→15″网格。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize
from matplotlib.cm import ScalarMappable
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                     "axes.unicode_minus": False, "font.size": 7})
d = np.load(OUT / "validation_G_fields.npz")
lon, lat, ssv, val = d["lon"], d["lat"], d["ssv"], d["val"]
bathy = np.load(ROOT / "output_real/three_parts/ship_regions/G_Japan.npz")["bathy"]
z = ssv[:ssv.shape[0]//4*4, :ssv.shape[1]//4*4]
blk = z.reshape(z.shape[0]//4, 4, z.shape[1]//4, 4).mean(axis=(1, 3))
blon = lon[::4][:blk.shape[1]]; blat = lat[::4][:blk.shape[0]]

# SDUST 1' prior on the same 1' grid
ds = xr.open_dataset(ROOT / "data/SDUST2022/SDUST2022GRA.nc")
sub = ds["ga"].sel(lon=slice(lon[0], lon[-1]), lat=slice(lat[0], lat[-1])).values
glon = np.asarray(ds.lon.sel(lon=slice(lon[0], lon[-1])))
glat = np.asarray(ds.lat.sel(lat=slice(lat[0], lat[-1])))
ds.close()
if glat[0] > glat[-1]:
    sub = sub[::-1]; glat = glat[::-1]
fi = RegularGridInterpolator((glat, glon), sub, bounds_error=False, fill_value=np.nan)
BLON, BLAT = np.meshgrid(blon, blat)
sdust = fi(np.column_stack([BLAT.ravel(), BLON.ravel()])).reshape(BLAT.shape)
sdust = np.where(np.isfinite(sdust), sdust, np.nanmedian(sdust))

step = 5
fig = plt.figure(figsize=(13.4, 5.4), dpi=220)
pos = [(0.03, 0.53, 0.27, 0.36), (0.36, 0.53, 0.27, 0.36), (0.69, 0.53, 0.27, 0.36),
       (0.03, 0.08, 0.44, 0.36), (0.55, 0.08, 0.40, 0.36)]
ext = [blon[0], blon[-1], blat[0], blat[-1]]

def map_ax(pos, title):
    ax = fig.add_axes(pos)
    ax.set_title(title, fontsize=7.8)
    ax.set_xticks([]); ax.set_yticks([])
    return ax

# (a) SDUST prior
ax = map_ax(pos[0], "(a) SDUST2022GRA：1′ 平滑先验")
v = np.nanpercentile(np.abs(sdust), 98)
levels = np.linspace(-v, v, 17)
cf = ax.contourf(BLON, BLAT, sdust, levels=levels, cmap="viridis", extend="both")
ax.contour(BLON, BLAT, sdust, levels=levels[::2], colors="k", linewidths=0.18, alpha=0.45)
cb = fig.colorbar(cf, ax=ax, fraction=0.045, pad=0.01)
cb.set_label("mGal", fontsize=5.5); cb.ax.tick_params(labelsize=4.5)

# (b) SSV block observation
ax = map_ax(pos[1], "(b) SSV33.1：1′ 块状观测")
v = np.nanpercentile(np.abs(blk), 98)
im = ax.imshow(blk[::step, ::step], cmap="RdBu_r", origin="lower",
               extent=ext, vmin=-v, vmax=v, aspect="auto", interpolation="nearest")
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.01)
cb.set_label("mGal", fontsize=5.5); cb.ax.tick_params(labelsize=4.5)

# (c) ship
ax = map_ax(pos[2], "(c) 船测重力：叠地形")
lsb = LightSource(azdeg=315, altdeg=45)
bzb = bathy[::5, ::5]
ax.imshow(lsb.shade(bzb, cmap=plt.cm.Greys, vert_exag=0.4, blend_mode="overlay"),
          origin="lower", extent=[lon[0], lon[-1], lat[0], lat[-1]],
          aspect="auto", alpha=0.85)
sc = ax.scatter(val[:,0], val[:,1], c=val[:,2], s=0.7, cmap="RdYlBu_r",
                vmin=np.percentile(val[:,2],2), vmax=np.percentile(val[:,2],98),
                linewidths=0, alpha=0.95)
ax.set_xlim(lon[0], lon[-1]); ax.set_ylim(lat[0], lat[-1])
cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.01)
cb.set_label("mGal", fontsize=5.5); cb.ax.tick_params(labelsize=4.5)

# (d) terrain
ax = map_ax(pos[3], "(d) GEBCO：15″ 地形")
ls = LightSource(azdeg=315, altdeg=45)
bz = bathy[::5, ::5]
ax.imshow(ls.shade(bz, cmap=plt.cm.Greys, vert_exag=0.4, blend_mode="overlay"),
          origin="lower", extent=[lon[0], lon[-1], lat[0], lat[-1]], aspect="auto")
sm = ScalarMappable(norm=Normalize(vmin=np.nanmin(bz), vmax=np.nanmax(bz)), cmap="Greys")
cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.01)
cb.set_label("水深 (m)", fontsize=5.5); cb.ax.tick_params(labelsize=4.5)

# (e) 1' -> 15"
ax = fig.add_axes(pos[4])
ax.set_xlim(0, 4.5); ax.set_ylim(-0.8, 4.5); ax.set_aspect("equal"); ax.axis("off")
for i in range(5):
    ax.plot([0.2, 4.2], [0.2+i, 0.2+i], color="#90A4AE", lw=0.7)
    ax.plot([0.2+i, 0.2+i], [0.2, 4.2], color="#90A4AE", lw=0.7)
ax.add_patch(plt.Rectangle((0.2, 0.2), 4, 4, fill=False, ec="#C0392B", lw=1.6))
for i in range(4):
    for j in range(4):
        ax.fill_between([0.2+i,0.2+i+1],[0.2+j,0.2+j],[0.2+j+1,0.2+j+1],
                        color="#EEF3F9", alpha=0.7, zorder=0)
ax.annotate("", xy=(4.2, -0.05), xytext=(0.2, -0.05),
            arrowprops=dict(arrowstyle="<->", color="#0B2545", lw=1.0))
ax.text(2.2, -0.50, "1′ 块 = 4×4 个 15″ 格点（15″ ≈ 0.46 km）", ha="center", fontsize=6.5)
ax.text(2.2, 2.1, "1′", ha="center", va="center", fontsize=18, color="#C0392B", alpha=0.75)
ax.set_title("(e) 1′ → 15″ 格网关系", fontsize=7.8)

fig.savefig(OUT / "fig_data_foundation.png", bbox_inches="tight")
print("saved")

