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
     "重力场是质量分布与自转离心力的叠加，不同波长对应不同深度的密度异常；"
     "短波主要来自海底地形和断裂，是海洋应用最关心的部分。",
     NEW / "fig_new_concept.png"),
    ("为什么要重建地球重力场",
     "1′ 产品在 5–20 km 被平滑，无法支撑 15″、1.85 km 的应用；"
     "重建是把这一频段的短波补回来，不是简单加密网格。",
     NEW / "fig_new_why.png"),
    ("现有的 1′ 地球重力场产品",
     "SSV33.1、DTU21GRA、SDUST2022GRA、NSOAS24 覆盖全球或中国近海，分辨率多为 1′；"
     "共同问题是 20 km 以下能量不足，直接插值补不回来。",
     NEW / "fig_new_products_mixed.png"),
    ("常用的 15″ 重力场重建方法",
     "插值、频域融合、深度学习和卡尔曼滤波是四条常见路线；"
     "前两者依赖插值核或波长先验，后两者依赖样本或协方差模型。",
     NEW / "fig_new_methods_2x2.png"),
    ("这些方法存在的缺点",
     "插值平滑短波，频域融合受先验和边界影响，深度学习缺物理约束，"
     "标准 KF 受协方差近似限制；5–20 km 难以兼顾精度与结构。",
     NEW / "fig_new_cons_mixed.png"),
    ("标准 KF 可以解决上述问题",
     "标准 KF 把重力场写成状态、卫星和船测写成观测，用预测—增益—更新递推；"
     "R/Q 自动分配权重，并给出逐像元方差。",
     NEW / "fig_new_stdkf_2x2.png"),
    ("标准 KF 存在的问题",
     "直接套用标准 KF 会遇到三个问题：对角 P 假设格点独立、1′ 块平均被当成点观测、"
     "船测更新后出现条带；固定噪声也难以适应不同海域。",
     NEW / "fig_new_stdkf_problems_2x2.png"),
    ("本项目 KF 重建过程",
     "以 1′ 和船测为输入，先验加入地形高频；6 轮块观测更新后，修正量集中在测线与构造带。"
     "这样做的原因是让修正沿构造扩散，而不是全局重算。",
     NEW / "fig_new_iterations_mixed.png"),
    ("本项目 KF 流程",
     "卫星、船测和地形先进入同一套 15″ 状态，经相干保持预测和 RTS 平滑输出重力场；"
     "再在 5–20 km 上做带通、THG 和细化，得到线性体。这样同一套场既能验证，也能直接进入断裂提取。",
     NEW / "fig_new_flow_fault.png"),
    ("验证",
     "三个有船测的区域按整条航次留出，训练时不使用该航次；"
     "检验的是对未知测线的预测能力，而不是对已参与建模数据的拟合。",
     NEW / "fig_new_validation_mixed.png"),
    ("精度总结",
     "南海-琉球和日本综合区改善 8.7% 和 6.3%，菲律宾海基本持平；"
     "留出航次说明 15″ 场在部分区域有稳定增益，也暴露了区域差异。",
     NEW / "fig_new_accuracy_rich.png"),
    ("断层",
     "在 15″ 场上做 5–20 km 带通和 THG，提取线性体并与 GM9 对比；"
     "为保住脊线，预测步用相干加权，而不是各向同性平滑。",
     NEW / "fig_new_fault_2x3.png"),
    ("总结",
     "相干保持 KF 把 1′、船测和地形融成 15″ 场，三区改善 0.2–8.7%，"
     "并支撑 5 km 覆盖 67.9% 的断裂识别；精度与结构因此能同时保留。",
     NEW / "fig_new_summary_rich.png"),
]

for i, (title, body, fig) in enumerate(pages, 1):
    add_page(title, body, fig, i)

prs.save(OUT)
print("saved", OUT, "pages", len(prs.slides))
