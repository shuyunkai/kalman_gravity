"""为“尺度分解/断裂识别”和“反演”准备一个测试区的 15″ 重力场。

用现有 KF 流程（BathyPriorKF, alpha=0.5, 6 轮）重建，缓存为 npz：
  output_real/test_field_15s.npz
    g15   : (ny, nx) 15″ 重力异常 (mGal)
    bathy : (ny, nx) GEBCO 15″ 水深 (m, 负值)
    lon, lat : 像元中心坐标
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from three_method_compare import load_region_data, build_three_methods, METHOD_KF

OUT = Path(__file__).parent.parent / "output_real" / "test_field_15s.npz"

# 日本海沟—伊豆-小笠原北部：构造最剧烈、船测较多
LON = (141.0, 149.0)
LAT = (32.0, 40.0)


def main():
    print(f"region lon={LON} lat={LAT}")
    data = load_region_data(LON, LAT, val_cruise="MR22-04_gravity_dmo", upscale=4)
    data["lon_range"] = LON
    data["lat_range"] = LAT
    print("building KF ...")
    models, times = build_three_methods(data, alpha=0.5)
    g15 = models[METHOD_KF]
    np.savez_compressed(
        OUT,
        g15=g15.astype(np.float32),
        bathy=data["bathy_15s"].astype(np.float32),
        lon=data["lon_hr"].astype(np.float64),
        lat=data["lat_hr"].astype(np.float64),
    )
    print(f"saved {OUT}  shape={g15.shape}  {OUT.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
