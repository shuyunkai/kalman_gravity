import sys, numpy as np, json
sys.path.insert(0,'src')
from three_method_compare import load_region_data
LON=(141.0,149.0); LAT=(32.0,40.0)
data=load_region_data(LON,LAT,val_cruise='MR22-04_gravity_dmo',upscale=4)
print('shapes',data['gravity_lr'].shape,data['bathy_15s'].shape)
print('train',len(data['train_points']),'val',len(data['val_points']))
print('val bounds',data['val_points'][:,0].min(),data['val_points'][:,0].max(),data['val_points'][:,1].min(),data['val_points'][:,1].max())
np.savez_compressed('output_real/three_parts/abc_region_inputs.npz', **{k:data[k] for k in ['gravity_lr','lon_lr','lat_lr','lon_hr','lat_hr','ship_grav','train_mask','bathy_15s','ssv_hr','train_points','val_points']})
print('saved')
