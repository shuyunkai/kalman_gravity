"""用 Earthdata Token 直接下载 SWOT L2_LR_SSH（不经 PO.DAAC 网站，走 CMR + 归档主机）。

Token 获取：https://urs.earthdata.nasa.gov/profile → Generate Token
保存到 D:\\gravity\\.edl_token（一行），或设置环境变量 EARTHDATA_TOKEN。

用法：
  python src/download_swot_token.py --bbox 141 32 149 40 \
      --start 2024-01-01 --end 2024-03-31 --limit 2
"""
import argparse
import os
import sys
import time
from pathlib import Path

import requests

CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"


def get_token():
    tok = os.environ.get("EARTHDATA_TOKEN")
    if tok:
        return tok.strip()
    p = Path(r"D:\gravity\.edl_token")
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    sys.exit("未找到 token：请生成后保存到 D:\\gravity\\.edl_token")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float, default=[141, 32, 149, 40])
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2024-03-31")
    ap.add_argument("--short-name", default="SWOT_L2_LR_SSH_D")
    ap.add_argument("--out", default=r"D:\gravity\kalman_gravity\data\swot")
    ap.add_argument("--limit", type=int, default=2)
    args = ap.parse_args()

    token = get_token()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {token}"}

    params = {
        "short_name": args.short_name,
        "bounding_box": ",".join(str(v) for v in args.bbox),
        "temporal": f"{args.start}T00:00:00Z,{args.end}T23:59:59Z",
        "page_size": 100,
    }
    r = requests.get(CMR, params=params, timeout=60)
    r.raise_for_status()
    entries = r.json()["feed"]["entry"]
    print("granules found:", len(entries))

    links = []
    for e in entries:
        for l in e.get("links", []):
            href = l.get("href", "")
            if l.get("rel", "").endswith("/data#") and href.endswith(".nc"):
                links.append(href)
    print("netcdf links:", len(links))

    n = 0
    for url in links:
        name = Path(url).name
        dst = out / name
        if dst.exists() and dst.stat().st_size > 1e6:
            print("skip", name)
            continue
        print("download", name)
        with requests.get(url, headers=headers, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            tmp = dst.with_suffix(".part")
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(1 << 20):
                    if chunk:
                        fh.write(chunk)
            tmp.replace(dst)
        n += 1
        print("  size MB", round(dst.stat().st_size / 1e6, 1))
        if n >= args.limit:
            break
        time.sleep(1)
    print("done")


if __name__ == "__main__":
    main()
