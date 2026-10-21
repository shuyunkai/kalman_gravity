"""SWOT L2 SSH 批量下载（首次运行弹出 Earthdata 登录，之后自动复用）。

用法：
  pip install earthaccess
  python src/download_swot.py --bbox 141 32 149 40 --start 2023-08-01 --end 2024-06-30

说明：
  - 需要免费 Earthdata 账号：https://urs.earthdata.nasa.gov/users/new
  - 首次会打开浏览器让你登录并授权 PO.DAAC；凭据保存在本机 ~/.netrc
  - 先下 1–3 个月试水，单个 granule 约几百 MB
"""
import argparse

import earthaccess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float,
                    default=[141.0, 32.0, 149.0, 40.0],
                    metavar=("LON0", "LAT0", "LON1", "LAT1"))
    ap.add_argument("--start", default="2023-08-01")
    ap.add_argument("--end", default="2024-06-30")
    ap.add_argument("--out", default=r"D:\gravity\kalman_gravity\data\swot")
    ap.add_argument("--short-name", default="SWOT_L2_LR_SSH_EXPERT")
    args = ap.parse_args()

    earthaccess.login(strategy="interactive", persist=True)
    res = earthaccess.search_data(
        short_name=args.short_name,
        bounding_box=tuple(args.bbox),
        temporal=(args.start, args.end),
    )
    print(f"granules: {len(res)}")
    if not res:
        print("没有结果：检查 short_name / 时间范围 / bbox")
        return
    earthaccess.download(res, args.out)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
