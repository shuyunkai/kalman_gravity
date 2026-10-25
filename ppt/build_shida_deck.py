"""以石大模板第 4 页为母版，生成 13 页。"""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).parent.parent
PPT_SRC = ROOT / "ppt" / "_石大空白_13页.pptx"
PPT_OUT = ROOT / "ppt" / "汇报_石大模板_13页.pptx"
NEW = ROOT / "output_real" / "three_parts" / "newdeck"

GOLD = RGBColor(0xC9, 0xA2, 0x27)
NAVY = RGBColor(0x0B, 0x1F, 0x3A)
CREAM = RGBColor(0xFB, 0xF7, 0xEA)
INK = RGBColor(0x1A, 0x1A, 0x1A)

prs = Presentation(PPT_SRC)

PAGES = [
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
     "直接套用会遇到三个问题：对角 P 假设格点独立、1′ 块平均被当成点观测、"
     "船测更新后出现条带；固定噪声也难适应不同海域。",
     NEW / "fig_new_stdkf_problems_2x2.png"),
    ("本项目 KF 重建过程",
     "以 1′ 和船测为输入，先验加入地形高频；6 轮块观测更新后，修正量集中在测线与构造带。"
     "这样做的原因是让修正沿构造扩散，而不是全局重算。",
     NEW / "fig_new_iterations_mixed.png"),
    ("本项目 KF 流程",
     "卫星、船测和地形先进入同一套 15″ 状态，经相干保持预测和 RTS 平滑输出重力场；"
     "再在 5–20 km 上做带通、THG 和细化，得到线性体。",
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
     "在 15″ 场上做 5–20 km 带通与 THG，提取线性体并与 GM9 对比；"
     "为保住脊线，预测步采用相干加权。",
     NEW / "fig_new_fault_2x3.png"),
    ("总结",
     "相干保持 KF 把 1′、船测和地形融成 15″ 场，三区改善 0.2–8.7%，"
     "并支撑 5 km 覆盖 67.9% 的断裂识别；精度与结构因此能同时保留。",
     NEW / "fig_new_summary_rich.png"),
]

NUMS = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二", "十三"]


def clear(slide):
    for sh in list(slide.shapes):
        top = sh.top / 914400 if sh.top is not None else 0
        name = sh.name
        if top < 1.0:
            if sh.shape_type == 17 and sh.width and sh.width / 914400 > 3:
                sh._element.getparent().remove(sh._element)
            continue
        if top > 7.0:
            continue
        sh._element.getparent().remove(sh._element)


def add_text(slide, x, y, w, h, text, size, color, bold=True,
             align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = 0; tf.margin_right = 0
    tf.margin_top = 0; tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = align; p.line_spacing = 1.25
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.bold = bold
    r.font.color.rgb = color; r.font.name = "Microsoft YaHei"
    return tb


for i, (title, body, fig) in enumerate(PAGES):
    s = prs.slides[i]
    clear(s)

    add_text(s, 0.92, 0.08, 10.6, 0.62,
             f"{NUMS[i]}、{title}", 25, GOLD)

    box = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.34), Inches(0.98),
                             Inches(12.66), Inches(1.26))
    box.fill.solid(); box.fill.fore_color.rgb = CREAM
    box.line.color.rgb = GOLD; box.line.width = Pt(0.9)
    tf = box.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.28); tf.margin_right = Inches(0.28)
    tf.margin_top = Inches(0.10); tf.margin_bottom = Inches(0.08)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.line_spacing = 1.35; p.alignment = PP_ALIGN.LEFT
    r = p.add_run(); r.text = body
    r.font.size = Pt(15.5); r.font.color.rgb = INK
    r.font.name = "Microsoft YaHei"

    img_w, img_h = Image.open(fig).size
    max_w, max_h = 12.5, 4.12
    w = max_w; h = w * img_h / img_w
    if h > max_h:
        h = max_h; w = h * img_w / img_h
    x = (13.333 - w) / 2
    y = 2.40 + (max_h - h) / 2
    s.shapes.add_picture(str(fig), Inches(x), Inches(y), width=Inches(w))

    add_text(s, 12.05, 7.10, 1.0, 0.28, f"{i+1} / 13", 9,
             RGBColor(0xD9, 0xC8, 0x9A), bold=False, align=PP_ALIGN.RIGHT)

prs.save(PPT_OUT)
print("saved", PPT_OUT, len(prs.slides))
