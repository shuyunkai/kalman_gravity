import sys; sys.path.insert(0,"src")
import numpy as np
from scipy.ndimage import gaussian_filter
from three_method_compare import load_region_data
for c in ("MR22-04_gravity_dmo","MR23-02_gravity_dmo","MR21-02_gravity_dmo"):
    d=load_region_data((141,149),(32,40),val_cruise=c,upscale=4)
    g=d["ship_grav"]; m=d["train_mask"]
    fill=np.where(m,g,0.0)
    sm=gaussian_filter(fill,20)/np.maximum(gaussian_filter(m.astype(float),20),1e-6)
    r=(g-sm)[m]
    mad=1.4826*np.median(np.abs(r-np.median(r)))
    print(c, "n",int(m.sum()), "std",round(float(np.std(r)),2),"MAD",round(float(mad),2))
