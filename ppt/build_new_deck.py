"""按新叙事重建 13 页：单标题 + 一段正文 + 图。"""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).parent.parent
NEW = ROOT / "output_real" / "three_parts" / "newdeck"
OLD = ROOT / "output_real" / "three_parts"
OUT = Path(__file__).parent / "汇报_新结构_13页.pptx"

NAVY = RGBColor(0x0B, 0x25, 0x45)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PALE = RGBColor(0xC9, 0xD9, 0xEA)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)


def add_page(title, body, fig, page):
    s = prs.slides.add_slide(prs.slide_layouts[6])

    from pptx.enum.shapes import MSO_SHAPE
    rect = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0,
                              prs.slide_width, Inches(2.30))
    rect.fill.solid(); rect.fill.fore_color.rgb = NAVY
    rect.line.fill.background()

    tb = s.shapes.add_textbox(Inches(0.7), Inches(0.28),
                              Inches(11.9), Inches(0.55))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = 0; tf.margin_right = 0
    tf.margin_top = 0; tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.line_spacing = 1.1
    r = p.add_run(); r.text = title
    r.font.size = Pt(28); r.font.bold = True
    r.font.color.rgb = WHITE; r.font.name = "Microsoft YaHei"

    tb = s.shapes.add_textbox(Inches(0.7), Inches(0.88),
                              Inches(11.9), Inches(1.5))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = 0; tf.margin_right = 0
    tf.margin_top = 0; tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.TOP
    p = tf.paragraphs[0]; p.line_spacing = 1.45
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run(); r.text = body
    r.font.size = Pt(17); r.font.color.rgb = PALE
    r.font.name = "Microsoft YaHei"

    # 页码：最下方小字
    tb = s.shapes.add_textbox(Inches(11.9), Inches(7.30),
                              Inches(0.8), Inches(0.18))
    tf = tb.text_frame; tf.margin_top = 0; tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.RIGHT
    r = p.add_run(); r.text = f"{page} / 13"
    r.font.size = Pt(8); r.font.color.rgb = RGBColor(0x5B, 0x65, 0x73)
    r.font.name = "Microsoft YaHei"

    # 图片自适应
    iw, ih = Image.open(fig).size
    max_w, max_h = 12.4, 4.78
    w = max_w; h = w * ih / iw
    if h > max_h:
        h = max_h; w = h * iw / ih
    x = (13.333 - w) / 2
    s.shapes.add_picture(str(fig), Inches(x), Inches(2.42),
                         width=Inches(w))


pages = [
    ("什么是地球重力场",
     "地球重力场是地球质量分布产生的引力与自转离心力的合力场；其异常反映地壳密度、"
     "海底地形与深部结构，不同波长对应不同深度的质量异常。",
     NEW / "fig_new_concept.png"),
    ("为什么要重建地球重力场",
     "现有 1′ 产品在 5–20 km 被平滑，海底地形、断裂与工程应用需要 15″、1.85 km 的"
     "高分辨率重力场；短波信息是海洋测绘与资源环境调查的关键。",
     NEW / "fig_new_why.png"),
    ("现有的 1′ 地球重力场产品",
     "SSV33.1、DTU21GRA、SDUST2022GRA、NSOAS24 等产品分辨率多为 1′；在三个验证区，"
     "15″ 重建相对 1′ 产品的 RMSE 改善 0.2%–8.7%。",
     NEW / "fig_new_products_mixed.png"),
    ("常用的 15″ 重力场重建方法",
     "常用方法包括插值、频域融合、深度学习与卡尔曼滤波；它们依赖插值核、波长先验、"
     "训练样本或协方差模型，目标是恢复 15″、5–20 km 的可信结构。",
     NEW / "fig_new_methods.png"),
    ("这些方法存在的缺点",
     "插值会平滑短波，频域融合依赖先验且受边界影响，深度学习缺物理约束与样本，"
     "标准 KF 则受协方差建模限制；1′ 产品在 20 km 以下的能量明显缺失。",
     NEW / "fig_new_cons_mixed.png"),
    ("标准 KF 可以解决上述问题",
     "标准卡尔曼滤波把重力场写成状态，把卫星、船测写成观测，通过预测—增益—更新递推；"
     "权重由 R/Q 自动决定，并能给出逐像元不确定度，适合多源融合。",
     NEW / "fig_new_stdkf.png"),
    ("标准 KF 存在的问题",
     "直接套用标准 KF 会遇到三个问题：对角 P 假设格点独立、1′ 块平均被当成点观测、"
     "船测更新后出现条带；固定噪声也难以适应不同海域。",
     NEW / "fig_new_stdkf_problems.png"),
    ("本项目 KF 重建过程",
     "以 1′ 产品和船测为输入，先验加入地形高频；经 6 轮块观测更新，"
     "修正量集中在测线与构造带，最终输出 15″ 重力场。",
     NEW / "fig_new_iterations_mixed.png"),
    ("本项目 KF 流程",
     "卫星 1′、船测与地形先验进入同一套 15″ 状态，经块观测与相干保持预测"
     "迭代 6 轮，再做 RTS 平滑，输出重力场与逐像元方差。",
     NEW / "fig_new_flow_fault.png"),
    ("验证",
     "三个有船测的 15″ 区域用整条留出航次验证：南海-琉球 5.34→4.88，菲律宾海 1.49→1.48，"
     "日本综合区 3.96→3.71（mGal）；相干保持 KF 在南海与日本更优，菲律宾海与标准 KF 持平。",
     NEW / "fig_new_validation_mixed.png"),
    ("精度总结",
     "相对 1′ 产品，南海-琉球改善 8.7%、日本 6.3%、菲律宾海基本持平；"
     "留出船测精度 3.59/4.84 mGal，说明 15″ 场在部分区域有稳定增益。",
     NEW / "fig_new_accuracy_rich.png"),
    ("断层",
     "在 15″ 场上做 5–20 km 带通与 THG；采用相干保持卡尔曼滤波，在预测步保留脊线细节，"
     "GM9 5 km 覆盖 67.9%、命中 44.5%，同时留出重力 RMSE 3.685 mGal，与普通 KF 的 3.681 基本持平。",
     NEW / "fig_new_fault_aniso.png"),
    ("总结",
     "相干保持卡尔曼滤波把 15″ 重力场重建为 1.85 km 网格，三区改善 0.2%–8.7%，"
     "并支撑 5 km 覆盖 67.9% 的断裂识别；方法、精度与应用形成闭环。",
     NEW / "fig_new_summary_rich.png"),
]

for i, (title, body, fig) in enumerate(pages, 1):
    add_page(title, body, fig, i)

prs.save(OUT)
print("saved", OUT, "pages", len(prs.slides))
