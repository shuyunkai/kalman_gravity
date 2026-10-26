"""以石大模板第 4 页为母版，生成 13 页。"""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).parent.parent
PPT_SRC = ROOT / "ppt" / "_石大空白_14页.pptx"
PPT_OUT = ROOT / "ppt" / "汇报_石大模板_14页.pptx"
NEW = ROOT / "output_real" / "three_parts" / "newdeck"

GOLD = RGBColor(0xC9, 0xA2, 0x27)
NAVY = RGBColor(0x0B, 0x1F, 0x3A)
CREAM = RGBColor(0xFB, 0xF7, 0xEA)
INK = RGBColor(0x1A, 0x1A, 0x1A)
RED = RGBColor(0xC0, 0x39, 0x2B)

prs = Presentation(PPT_SRC)

PAGES = [
    ("什么是地球重力场",
     "地球重力场重建的是重力异常，即实际重力 g 与参考椭球正常重力 γ 之差 Δg = g − γ；"
     "它反映地下质量分布，不同波长对应不同深度，短波主要来自海底地形与断裂。",
     NEW / "fig_new_concept_rich.png"),
    ("为什么要重建地球重力场",
     "重力辅助导航依赖短波梯度，海底地形与工程依赖 5–20 km 结构，资源勘查依赖布格异常，"
     "深部结构依赖长波；15″ 重建是为了让这些应用各自拿到可用的波段，而不是单纯提高分辨率。",
     NEW / "fig_new_why_matrix.png"),
    ("现有的 1′ 地球重力场产品",
     "SSV33.1、DTU21GRA、SDUST2022GRA、NSOAS24 都是全球海洋重力产品，分辨率多为 1′；"
     "它们能提供长波和中波，但 20 km 以下能量不足，产品之间也有差异，因此不能直接当作 15″ 场使用。",
     NEW / "fig_new_products_matrix.png"),
    ("常用的 15″ 重力场重建方法",
     "插值、频域融合、最小二乘配置和多源迭代是四条常见路线；"
     "插值靠核函数，频域融合靠波长先验，LSC 靠协方差函数，迭代靠初值与权重，目标都是把 1′ 补到 15″。",
     NEW / "fig_new_methods_4lanes.png"),
    ("这些方法存在的缺点",
     "插值只能平滑，频域融合在边界振铃，LSC 对协方差函数和计算量敏感，多源迭代依赖初值与块边界；"
     "四者都缺少对观测噪声和空间相关的显式描述，5–20 km 的结构难以稳定恢复。",
     NEW / "fig_new_cons_4methods.png"),
    ("标准 KF 可以解决上述问题",
     "标准 KF 把重力场写成状态、卫星和船测写成观测，用预测—增益—更新递推；"
     "它按 R/Q 自动分配权重，并给出逐像元方差，因此适合把不同来源、不同精度的数据放进同一套估计。",
     NEW / "fig_new_stdkf_cycle.png"),
    ("标准 KF 存在的问题",
     "直接套用会遇到三个问题：对角 P 假设格点独立、1′ 块平均被当成点观测、船测更新后出现条带；"
     "单点更新只改一个像素，邻域没有一起修正，所以标准 KF 不能直接拿来重建 15″ 场。",
     NEW / "fig_new_stdkf_problems_2x2.png"),
    ("本项目提出的相干保持 KF",
     "本项目提出相干保持卡尔曼滤波：在标准 KF 的预测步引入结构张量相干 C，"
     "用 x⁻=(1−wC²)Gσx+wC²x 让脊线处少平滑、平坦区正常平滑。"
     "它主要解决标准 KF 把 5–20 km 相干短波磨平的问题，在不牺牲重力 RMSE 的前提下保住断裂与地形结构。",
     NEW / "fig_new_contribution_formula.png"),
    ("本项目 KF 流程",
     "卫星 1′、船测和地形先进入同一套 15″ 状态：块平均得到 1′ 观测，地形高频构造先验；"
     "初始化后按“预测—卫星块更新—船测更新”迭代 6 轮，再做 RTS 平滑，输出 15″ 场与逐像元方差。",
     NEW / "fig_new_kf_flow_full.png"),
    ("本项目 KF 重建过程",
     "以 1′ 和船测为输入，先验加入地形高频，经 6 轮块观测更新，修正量集中在测线与构造带；"
     "这样做的原因是让修正沿构造扩散，而不是全局重算，从而在保住长波的同时补出短波。",
     NEW / "fig_new_iterations_mixed.png"),
    ("验证",
     "三个有船测的区域按整条航次留出，训练时不使用该航次；"
     "这样检验的是对未知测线的预测能力，而不是对已参与建模数据的拟合，结果才能说明 15″ 场在真实海域是否可用。",
     NEW / "fig_new_validation_zoom.png"),
    ("精度总结",
     "南海-琉球和日本相对标准 KF 改善 1.4% 和 4.4%，相对 1′ 为 8.7% 和 6.3%；"
     "菲律宾海反而差 0.2%，不算增益。",
     NEW / "fig_new_accuracy_table.png"),
    ("断层",
     "在 15″ 场上做 5–20 km 带通和 THG，提取线性体并与 GM9 对比；"
     "为保住脊线，预测步采用相干加权，而不是各向同性平滑，这样短波结构不会被磨掉。",
     NEW / "fig_new_fault_2x3.png"),
    ("总结",
     "相干保持 KF 把 1′、船测和地形融成 15″ 场，南海/日本相对标准 KF 改善 1.4%/4.4%，"
     "并支撑 5 km 覆盖 67.9% 的断裂识别；菲律宾海略差 0.2%，不计入增益。",
     NEW / "fig_new_summary_synthesis.png"),
]

HIGHLIGHTS = [
    ["重力异常", "Δg = g − γ", "5–20 km"],
    ["短波梯度", "5–20 km", "可用的波段"],
    ["1′", "20 km 以下", "不能满足"],
    ["四条常见路线", "核函数", "波长先验", "协方差函数", "初值与权重"],
    ["只平滑", "边界振铃", "协方差函数", "块边界", "5–20 km"],
    ["状态", "观测", "预测—增益—更新", "逐像元方差"],
    ["对角 P", "块平均", "条带", "单点更新"],
    ["相干保持卡尔曼滤波", "结构张量相干 C", "5–20 km", "不牺牲重力 RMSE"],
    ["1′ 观测", "地形高频", "6 轮", "RTS", "逐像元方差"],
    ["地形高频", "6 轮", "测线与构造带", "沿构造扩散"],
    ["整条航次留出", "未知测线"],
    ["1.4%", "4.4%", "8.7%", "6.3%", "0.2%"],
    ["5–20 km", "THG", "相干加权"],
    ["1.4%", "4.4%", "67.9%", "0.2%"],
]

NUMS = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二", "十三", "十四"]


def clear(slide):
    for sh in list(slide.shapes):
        top = sh.top / 914400 if sh.top is not None else 0
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


def add_body(p, text, words, size):
    pos = 0
    for w in words:
        idx = text.find(w, pos)
        if idx < 0:
            continue
        if idx > pos:
            r = p.add_run(); r.text = text[pos:idx]
            r.font.size = Pt(size); r.font.bold = True
            r.font.color.rgb = INK; r.font.name = "Microsoft YaHei"
        r = p.add_run(); r.text = w
        r.font.size = Pt(size); r.font.bold = True
        r.font.color.rgb = RED; r.font.name = "Microsoft YaHei"
        pos = idx + len(w)
    if pos < len(text):
        r = p.add_run(); r.text = text[pos:]
        r.font.size = Pt(size); r.font.bold = True
        r.font.color.rgb = INK; r.font.name = "Microsoft YaHei"


for i, (title, body, fig) in enumerate(PAGES):
    s = prs.slides[i]
    clear(s)

    add_text(s, 0.92, 0.08, 10.6, 0.62,
             f"{NUMS[i]}、{title}", 26, GOLD)

    box = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.34), Inches(0.98),
                             Inches(12.66), Inches(1.26))
    box.fill.solid(); box.fill.fore_color.rgb = CREAM
    box.line.color.rgb = GOLD; box.line.width = Pt(0.9)
    tf = box.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.28); tf.margin_right = Inches(0.28)
    tf.margin_top = Inches(0.08); tf.margin_bottom = Inches(0.06)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.line_spacing = 1.25; p.alignment = PP_ALIGN.LEFT
    add_body(p, body, HIGHLIGHTS[i], 17)

    img_w, img_h = Image.open(fig).size
    max_w, max_h = 12.5, 4.12
    w = max_w; h = w * img_h / img_w
    if h > max_h:
        h = max_h; w = h * img_w / img_h
    x = (13.333 - w) / 2
    y = 2.40 + (max_h - h) / 2
    s.shapes.add_picture(str(fig), Inches(x), Inches(y), width=Inches(w))

    add_text(s, 12.05, 7.10, 1.0, 0.28, f"{i+1} / 14", 9,
             RGBColor(0xD9, 0xC8, 0x9A), bold=False, align=PP_ALIGN.RIGHT)

prs.save(PPT_OUT)
print("saved", PPT_OUT, len(prs.slides))
