"""下载 JODC MGD77 netCDF（含 gravity/Free/Bathymetry 变量，免解码）。"""
import json
import subprocess
from pathlib import Path

INV = Path(r"D:\gravity\mgd77_inventory.json")
OUT = Path(r"D:\gravity\kalman_gravity\data\ship_gravity\jodc_nc")
LON, LAT = (141.0, 149.0), (32.0, 40.0)


def main():
    d = json.loads(INV.read_text(encoding="utf-8"))
    sel = [r for r in d["INVENTORY"]
           if r.get("gravity_flag")
           and r["lon_max"] >= LON[0] and r["lon_min"] <= LON[1]
           and r["lat_max"] >= LAT[0] and r["lat_min"] <= LAT[1]]
    OUT.mkdir(parents=True, exist_ok=True)
    print("voyages", len(sel))
    n = 0
    for r in sel:
        v = r["voyage"]
        dst = OUT / f"{v}.nc"
        if dst.exists() and dst.stat().st_size > 10000:
            continue
        p = subprocess.run(["curl.exe", "-sSL", "--retry", "3", "--max-time", "300",
                            "-o", str(dst),
                            f"https://www.jodc.go.jp/vpage/mgd77/nc/{v}.nc"],
                           capture_output=True)
        if p.returncode == 0 and dst.exists():
            n += 1
    print("downloaded", n)


if __name__ == "__main__":
    main()
