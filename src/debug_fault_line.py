"""诊断大断裂为何未被 THG 脊线识别。"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import zoom as sp_zoom
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from run_pipeline import load_ssv_region
from three_method_compare import load_bathy_15s
from utils import cell_centers
from aniso_kf_test import build, LON, LAT
from scale_decomposition import bandpass, thg, lineament_mask, filter_lineaments

ROOT = Path(__file__).parent.parent
g_lr, _, _, _ = load_ssv_region(LON, LAT)
ny, nx = g_lr.shape; up = 4
lon = cell_centers(LON[0], LON[1], nx*up)
lat = cell_centers(LAT[0], LAT[1], ny*up)
bathy = load_bathy_15s(LON, LAT, lon, lat)
g = build(g_lr, bathy, lon, lat, w_keep=1.0)
g1 = sp_zoom(g_lr, (up, up), order=1)
dx = float(lon[1]-lon[0])*111320*np.cos(np.radians(lat.mean()))
dy = float(lat[1]-lat[0])*111320
b = bandpass(g, dx, dy, 5e3, 20e3)
t = thg(b, dx, dy)
mask = filter_lineaments(lineament_mask(b, dx, dy, pct=60), 20, 1.8)
b1 = bandpass(g1, dx, dy, 5e3, 20e3)
mask1 = filter_lineaments(lineament_mask(b1, dx, dy, pct=60), 20, 1.8)

# 红线近似折线（从图中读）
red = np.array([[170.70,-46.20],[170.72,-45.95],[170.78,-45.75],
                [170.90,-45.60],[171.05,-45.50],[171.25,-45.40],
                [171.45,-45.32],[171.60,-45.25]])
# densify
seg=[]; 
for a,bp in zip(red[:-1],red[1:]):
    n=max(2,int(np.hypot((bp[0]-a[0])*np.cos(np.radians(35)),bp[1]-a[1])/0.002))
    for s in np.linspace(0,1,n,endpoint=False):
        seg.append(a+(bp-a)*s)
seg.append(red[-1]); seg=np.array(seg)

def samp(fld):
    ix=np.clip(np.searchsorted(lon,seg[:,0]),0,len(lon)-1)
    iy=np.clip(np.searchsorted(lat,seg[:,1]),0,len(lat)-1)
    return fld[iy,ix]
tv=samp(t); mv=samp(mask); m1=samp(mask1)
print('THG沿红线 mGal/km?', np.percentile(tv,[0,25,50,75,100]))
print('THG global pct', np.percentile(t,[50,60,70,80,90,95,99]))
print('mask on line %', mv.mean()*100, 'mask1', m1.mean()*100)
print('mask total px', mask.sum(), 'mask1', mask1.sum())
# nearest mask distance
ys,xs=np.where(mask)
pd=np.column_stack([lon[xs],lat[ys]])
d=cKDTree(pd).query(seg)[0]*111.3
print('dist to mask km', np.percentile(d,[0,50,90,100]))
# non-bandpassed THG / longer band
t_raw=thg(g,dx,dy); tvr=samp(t_raw)
b_long=bandpass(g,dx,dy,10e3,50e3); tl=thg(b_long,dx,dy); tvl=samp(tl)
print('raw THG line pct',np.percentile(tvr,[50,90]),'longband THG',np.percentile(tvl,[50,90]))
print('raw THG global pct',np.percentile(t_raw,[50,60,70,80,90,95,99]))
mask_long=filter_lineaments(lineament_mask(b_long,dx,dy,pct=60),20,1.8)
for pp in (30,40,50):
    mm=filter_lineaments(lineament_mask(b,dx,dy,pct=pp),20,1.8)
    ys,xs=np.where(mm)
    dd=cKDTree(np.column_stack([lon[xs],lat[ys]])).query(seg)[0]*111.3
    print('pct',pp,'on%',round(float(samp(mm).mean()*100),2),
          'd50',round(float(np.median(dd)),2),'total',mm.sum())
b_vlong=bandpass(g,dx,dy,20e3,80e3)
print('20-80 THG',np.percentile(samp(thg(b_vlong,dx,dy)),[50,90]))
print('mask_long line%',samp(mask_long).mean()*100,'total',mask_long.sum())
ys,xs=np.where(mask_long)
pd=cKDTree(np.column_stack([lon[xs],lat[ys]])).query(seg)[0]*111.3
print('dist mask_long km',np.percentile(pd,[0,50,90,100]))
# GM9 nearest lines
lines=[]
for f in json.load(open(ROOT/'output_real/three_parts/nz_gm9_offshore.json',encoding='utf-8'))['features']:
    geo=f.get('geometry') or {}
    if geo.get('type')!='LineString': continue
    a=np.asarray(geo['coordinates'],float)
    if ((a[:,0]>=LON[0])&(a[:,0]<=LON[1])&(a[:,1]>=LAT[0])&(a[:,1]<=LAT[1])).any(): lines.append(a)
best=[]
for a in lines:
    dd=cKDTree(a).query(red)[0]*111.3
    best.append((dd.mean(), len(a), a))
best.sort(key=lambda z:z[0])
print('nearest GM9 lines mean km',[round(x[0],2) for x in best[:5]], 'nlines',len(lines))
