"""用 PB2002 板块边界（含海沟/转换断层）作为公开构造线基准，按 2/5/10 km 缓冲验证。"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from scale_decomposition import bandpass, lineament_mask, filter_lineaments
from fault_validation import densify, fault_match

OUT = Path(__file__).parent.parent / "output_real" / "three_parts"


def load_pb2002(path, bbox):
    lon0, lat0, lon1, lat1 = bbox
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    lines = []
    for f in d["features"]:
        g = f.get("geometry") or {}
        if g.get("type") != "LineString":
            continue
        arr = np.asarray(g["coordinates"], float)
        inside = ((arr[:, 0] >= lon0) & (arr[:, 0] <= lon1) &
                  (arr[:, 1] >= lat0) & (arr[:, 1] <= lat1))
        if inside.any():
            lines.append(arr)
    return lines


def main():
    d = np.load(OUT.parent / "test_field_15s.npz")
    g = d["g15"].astype(float)
    lon, lat = d["lon"], d["lat"]
    dx = float(lon[1] - lon[0]) * 111320 * np.cos(np.radians(lat.mean()))
    dy = float(lat[1] - lat[0]) * 111320

    lines = load_pb2002(OUT / "pb2002.json", (141, 32, 149, 40))
    print("PB2002 lines in bbox", len(lines))
    blon, blat = densify(lines, 1.0)
    print("points", len(blon))

    b = bandpass(g, dx, dy, 10e3, 50e3)
    mask = filter_lineaments(lineament_mask(b, dx, dy, pct=84) & (b < 1e9),
                             min_area=60, min_elong=2.2)
    print("lineaments", mask.sum())
    for r in (2, 5, 10, 20, 50):
        st = fault_match(mask, blon, blat, lon, lat, radius_km=r)
        print(f"r={r:2d}: 覆盖 {st['recall']*100:5.1f}% (随机 {st['recall_random']*100:4.1f}%) "
              f"| 命中 {st['precision']*100:5.1f}% (随机 {st['precision_random']*100:4.1f}%)")


if __name__ == "__main__":
    main()
