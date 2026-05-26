"""统一命令行入口，调用区域管线、全球模型或图表生成。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="卡尔曼滤波重力场建模")
    parser.add_argument("--region", nargs=4, type=float,
                        default=[138, 152, 32, 48],
                        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
                        help="研究区域 (默认: 日本海沟)")
    parser.add_argument("--upscale", type=int, default=4,
                        help="超分辨率倍数 (默认: 4)")
    parser.add_argument("--global-model", action="store_true",
                        dest="global_model", help="构建全球重力场模型")
    parser.add_argument("--figures", action="store_true",
                        help="生成分析图表")
    parser.add_argument("--tile-size", type=float, default=20,
                        help="全球模型分块大小 (度)")
    args = parser.parse_args()

    if args.global_model:
        from run_global import create_global_gravity_model
        create_global_gravity_model(tile_size_lat=args.tile_size)
    elif args.figures:
        from make_figures import (fig1_global_gravity_map,
                                   fig2_shipborne_distribution,
                                   fig3_regional_zoom,
                                   fig4_scatter_and_histogram,
                                   fig5_power_spectrum,
                                   fig6_bathymetry_inversion,
                                   fig7_summary_table)
        for fig_func in [fig1_global_gravity_map, fig2_shipborne_distribution,
                         fig3_regional_zoom, fig4_scatter_and_histogram,
                         fig5_power_spectrum, fig6_bathymetry_inversion,
                         fig7_summary_table]:
            try:
                fig_func()
            except Exception as e:
                print(f"  Skipped {fig_func.__name__}: {e}")
    else:
        lon_min, lon_max, lat_min, lat_max = args.region
        from run_pipeline import run_pipeline
        run_pipeline(lon_range=(lon_min, lon_max),
                     lat_range=(lat_min, lat_max),
                     upscale=args.upscale)


if __name__ == "__main__":
    main()
