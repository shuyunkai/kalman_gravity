"""第15页：迭代场 + 修正量收敛曲线。"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0, str(Path(__file__).parent))
from three_method_compare import load_region_data
from aniso_kf_test import build
ROOT = Path(__file__).parent.parent
OUT = ROOT / "output_real/three_parts/newdeck"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False, "font.size": 10})
LON, LAT = (132.0, 152.0), (26.0, 42.0)
d = load_region_data(LON, LAT, val_cruise="MR22-04_gravity_dmo", upscale=4)
ds=xr.open_dataset(ROOT/'data/SDUST2022/SDUST2022GRA.nc')
sub=ds['ga'].sel(lon=slice(LON[0],LON[1]),lat=slice(LAT[0],LAT[1])).values
glon=np.asarray(ds.lon.sel(lon=slice(LON[0],LON[1]))); glat=np.asarray(ds.lat.sel(lat=slice(LAT[0],LAT[1]))); ds.close()
if glat[0]>glat[-1]: sub=sub[::-1]; glat=glat[::-1]
fi=RegularGridInterpolator((glat,glon),sub,bounds_error=False,fill_value=np.nan)
LLON,LLAT=np.meshgrid(d['lon_hr'],d['lat_hr'])
prior=fi(np.column_stack([LLAT.ravel(),LLON.ravel()])).reshape(LLAT.shape)
prior=np.where(np.isfinite(prior),prior,np.nanmedian(prior))
r = build(d["gravity_lr"].astype(float), d["bathy_15s"], d["lon_hr"], d["lat_hr"],
          w_keep=1.0, p0=2.0, n_iter=6, ship=d["ship_grav"], train=d["train_mask"],
          ssv_mode='obs_only', p0_obs=3.0, prior_long=prior,
          return_reconstructor=True)
lon, lat = d["lon_hr"], d["lat_hr"]
val = np.asarray(d["val_points"], float)
ext = [lon[0], lon[-1], lat[0], lat[-1]]
k = 4; sub = (slice(k, -k), slice(k, -k))
# correction norms between consecutive iterations
hist = [r.prior_2d] + [r.x_history[i] for i in range(len(r.x_history))]
corr = []
for i in range(1, len(hist)):
    diff = hist[i] - hist[i-1]
    corr.append(float(np.sqrt(np.nanmean(diff**2))))
print("CONV", corr)
np.savez(OUT / "kf_convergence.npz", corr=np.array(corr))
fields = [("(a) 输入：1′ + 船测", d["ssv_hr"]),
          ("(b) 先验：SDUST + 地形", r.prior_2d),
          ("(c) 第 1 轮", r.x_history[0]),
          ("(e) 第 6 轮", r.x_history[-1])]
v = np.nanpercentile(np.abs(r.x_history[-1][sub]), 98)
fig, axes = plt.subplots(2, 3, figsize=(13.4, 5.2), dpi=220)
for ax, (name, fld) in zip([axes[0,0],axes[0,1],axes[0,2],axes[1,1]], fields):
    im = ax.imshow(fld[sub], cmap="RdBu_r", origin="lower", vmin=-v, vmax=v, extent=ext, aspect="auto")
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("经度 (°E)", fontsize=8); ax.set_ylabel("纬度 (°N)", fontsize=8)
    ax.tick_params(labelsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("mGal", fontsize=7); cb.ax.tick_params(labelsize=6)
axes[0,0].scatter(val[::6,0], val[::6,1], s=0.5, c="k", alpha=0.5)
# (d) convergence
ax = axes[1,0]
iters = np.arange(1, len(corr)+1)
ax.plot(iters, corr, "o-", color="#C0392B", lw=1.8, ms=4)
for x,y in zip(iters,corr):
    ax.annotate(f"{y:.2f}", (x,y), textcoords="offset points", xytext=(0,6), ha="center", fontsize=6.5, color="#C0392B")
ax.set_title("(d) 修正量随轮次下降", fontsize=10)
ax.set_xlabel("迭代轮次", fontsize=8); ax.set_ylabel("修正量 RMS (mGal)", fontsize=8)
ax.set_xticks(iters); ax.grid(alpha=0.25); ax.tick_params(labelsize=7)
# (f) diff
ax = axes[1,2]
inc = r.x_history[-1] - r.x_history[0]
vi = np.nanpercentile(np.abs(inc[sub]), 98)
im = ax.imshow(inc[sub], cmap="PuOr_r", origin="lower", vmin=-vi, vmax=vi, extent=ext, aspect="auto")
ax.set_title("(f) 第 6 轮 − 第 1 轮", fontsize=10)
ax.set_xlabel("经度 (°E)", fontsize=8); ax.set_ylabel("纬度 (°N)", fontsize=8)
ax.tick_params(labelsize=7)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("mGal", fontsize=7); cb.ax.tick_params(labelsize=6)
fig.tight_layout()
p = OUT / "fig_new_iterations_conv.png"
fig.savefig(p, bbox_inches="tight"); plt.close(fig)
print("saved", p)
