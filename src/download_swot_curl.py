"""用 curl 下载 SWOT Basic（CMR 查询 + Bearer token），避开 requests 的连接重置。"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"


def token():
    tok = os.environ.get("EARTHDATA_TOKEN")
    if tok:
        return tok.strip()
    p = Path(r"D:\gravity\.edl_token")
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    sys.exit("missing token")


def curl(args, out=None):
    cmd = ["curl.exe", "-sSL", "--retry", "5", "--retry-delay", "3",
           "--max-time", "900"] + args
    if out:
        cmd += ["-o", str(out)]
    return subprocess.run(cmd, capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float, default=[141, 32, 149, 40])
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2024-03-31")
    ap.add_argument("--product", default="Basic")
    ap.add_argument("--out", default=r"D:\gravity\kalman_gravity\data\swot")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-pages", type=int, default=10)
    a = ap.parse_args()

    tok = token()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    base_params = {
        "short_name": "SWOT_L2_LR_SSH_D",
        "bounding_box": ",".join(str(v) for v in a.bbox),
        "temporal": f"{a.start}T00:00:00Z,{a.end}T23:59:59Z",
        "page_size": 200,
    }
    links = []
    for page in range(1, a.max_pages + 1):
        params = urlencode(dict(base_params, page_num=page))
        url = f"{CMR}?{params}"
        js = out / f"_cmr_{page}.json"
        for i in range(5):
            r = curl([url], js)
            if r.returncode == 0 and js.exists() and js.stat().st_size > 100:
                break
            print("cmr retry", page, i + 1, r.returncode)
            time.sleep(5 * (i + 1))
        else:
            break
        d = json.loads(js.read_text(encoding="utf-8"))
        entries = d["feed"]["entry"]
        print("page", page, "entries", len(entries))
        if not entries:
            break
        for e in entries:
            for l in e.get("links", []):
                h = l.get("href", "")
                if l.get("rel", "").endswith("/data#") and h.endswith(".nc") \
                        and f"_{a.product}_" in h:
                    links.append(h)
        if len(entries) < 200:
            break
    links = list(dict.fromkeys(links))
    print("links", len(links))

    n = 0
    for h in links:
        name = Path(h).name
        dst = out / name
        if dst.exists() and dst.stat().st_size > 1e6:
            print("skip", name); continue
        print("get", name)
        r = curl(["-H", f"Authorization: Bearer {tok}", h], dst)
        if r.returncode != 0 or not dst.exists():
            print("  fail", r.returncode); continue
        print("  MB", round(dst.stat().st_size / 1e6, 1))
        n += 1
        if n >= a.limit:
            break
        time.sleep(1)
    print("done", n)


if __name__ == "__main__":
    main()
