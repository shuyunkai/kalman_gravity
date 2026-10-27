"""以石大模板第 4 页为母版，生成 13 页。"""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).parent.parent
PPT_SRC = ROOT / "ppt" / "_石大空白_15页.pptx"
PPT_OUT = ROOT / "ppt" / "汇报_石大模板_15页.pptx"
NEW = ROOT / "output_real" / "three_parts" / "newdeck"

GOLD = RGBColor(0xC9, 0xA2, 0x27)
NAVY = RGBColor(0x0B, 0x1F, 0x3A)
CREAM = RGBColor(0xFB, 0xF7, 0xEA)
INK = RGBColor(0x1A, 0x1A, 0x1A)
RED = RGBColor(0xC0, 0x39, 0x2B)

prs = Presentation(PPT_SRC)

PAGES = [
    ("什么是地球重力场",
     "重建对象是重力异常 Δg = g − γ，而不是绝对重力；"
     "5–20 km 短波主要来自海底地形与断裂，是导航与工程直接使用的频段。",
     NEW / "fig_new_concept_rich.png"),
    ("为什么要重建地球重力场",
     "重力辅助导航依赖短波梯度，海底地形与工程依赖 5–20 km 结构，资源勘查依赖布格异常，"
     "深部结构依赖长波；15″ 重建是为了让这些应用各自拿到可用的波段，而不是单纯提高分辨率。",
     NEW / "fig_new_why_matrix.png"),
    ("现有的 1′ 地球重力场产品",
     "SSV33.1、DTU21GRA、SDUST2022GRA、NSOAS24 都是全球海洋重力产品，分辨率多为 1′；"
     "它们能提供长波和中波，但共同的不足是 20 km 以下能量弱、且没有逐像元方差，"
     "因此不能满足 15″ 应用。",
     NEW / "fig_new_products_matrix.png"),
    ("从 1′ 到 15″ 的常用重建方法",
     "插值、频域融合和最小二乘配置是三条路线；插值只把 1′ 加密，频域融合按谱权补短波，"
     "LSC 靠协方差预测，最终解算到 15″ 格点。",
     NEW / "fig_new_methods_3lanes.png"),
    ("这些方法存在的缺点",
     "插值只能平滑，频域融合在边界振铃，LSC 对协方差函数和计算量敏感；"
     "它们对噪声和短波的处理仍依赖人工选择，5–20 km 的结构难以稳定恢复。",
     NEW / "fig_new_cons_3methods.png"),
    ("标准 KF 可以解决上述问题",
     "插值会把 1′ 的短波磨平，KF 用卫星块和船测点的增益把短波补回；频域融合有边界振铃，"
     "KF 在空间域逐块递推，不涉及 FFT 边界；LSC 需要全局协方差求逆，KF 用块增益和 R/Q 做局部更新；"
     "但直接套用仍会遇到 P、块观测和船测条带的问题。",
     NEW / "fig_new_kf_mapping.png"),
    ("标准 KF 的流程",
     "标准 KF 把重力场写成状态、卫星和船测写成观测，用预测—增益—更新递推；"
     "它按 R/Q 自动分配权重，并给出逐像元方差，因此适合把不同来源、不同精度的数据放进同一套估计。",
     NEW / "fig_new_stdkf_cycle.png"),
    ("标准 KF 存在的问题",
     "从协方差看，直接套用会遇到四个问题：对角 P 把格点当成独立、1′ 块平均被当成点观测、"
     "船测更新后出现条带、单点更新只改一个像素；邻域没有一起修正，所以标准 KF 不能直接拿来重建 15″ 场。",
     NEW / "fig_new_stdkf_problems_2x2.png"),
    ("本项目提出的相干保持 KF",
     "针对对角 P 把格点当成独立、单点更新只改一个像素，相干保持 KF 用结构张量相干 C 构造空间相关的预测核："
     "x⁻=(1−wC²)Gσx+wC²x；预测时邻域一起参与，船测修正沿构造扩散，脊线处不被抹平，平坦区仍正常平滑。",
     NEW / "fig_new_contribution_formula.png"),
    ("本项目 KF 流程",
     "卫星 1′、船测和地形先进入同一套 15″ 状态：块平均得到 1′ 观测，地形高频构造先验；"
     "迭代 6 轮并做 RTS 平滑。这样做是为了让块观测、先验和船测在同一递推中分配权重。",
     NEW / "fig_new_kf_flow_full.png"),
    ("本项目 KF 重建过程",
     "以 1′ 和船测为输入，先验加入地形高频，经 6 轮块观测更新，修正量集中在测线与构造带；"
     "这样做的原因是让修正沿构造扩散，而不是全局重算，从而在保住长波的同时补出短波。",
     NEW / "fig_new_iterations_mixed.png"),
    ("精度总结",
     "三个区域都用整条航次留出，分别比较双线性 15″ 场与相干保持 KF；"
     "双线性场由 SSV33.1 的 1′ 双线性插值得到，局部差异见下页。",
     NEW / "fig_new_accuracy_table.png"),
    ("验证",
     "三个有船测的区域按整条航次留出，训练时不使用该航次；"
     "下图以日本区为例展示局部差异，三区统计见前一页。这样检验的是对未知测线的预测能力。",
     NEW / "fig_new_validation_zoom.png"),
    ("断层",
     "把 SSV33.1（1′）与相干保持 KF 场放在同一套 5–20 km 带通、THG 和 60% 分位阈值下提取线性体，"
     "再与 GM9 对比；这样比较的是输入场，而不是换方法。",
     NEW / "fig_new_fault_metrics.png"),
    ("总结",
     "把 1′、船测和地形融成同一套 15″ 场，并在三个留出航次和 GM9 断裂上检验；"
     "这样同一套场既能看重力精度，也能看短波结构是否保住。",
     NEW / "fig_new_summary_synthesis.png"),
]

HIGHLIGHTS = [
    ["重力异常", "Δg = g − γ", "5–20 km"],
    ["短波梯度", "5–20 km", "可用的波段"],
    ["1′", "20 km 以下", "不能满足"],
    ["三条常用路线", "谱权", "协方差", "15″ 格点"],
    ["只平滑", "边界振铃", "协方差函数", "5–20 km"],
    ["观测更新", "空间域递推", "块增益", "R/Q/P"],
    ["状态", "观测", "预测—增益—更新", "逐像元方差"],
    ["对角 P", "块平均", "条带", "单点更新"],
    ["相干保持卡尔曼滤波", "结构张量相干 C", "5–20 km", "不牺牲重力 RMSE"],
    ["1′ 观测", "地形高频", "6 轮", "RTS", "同一递推"],
    ["地形高频", "6 轮", "测线与构造带", "沿构造扩散"],
    ["整条航次留出", "常用 15″ 场", "相干保持 KF"],
    ["整条航次留出", "日本区", "未知测线"],
    ["SSV33.1", "相干保持 KF", "5–20 km", "THG"],
    ["15″ 场", "精度", "短波结构"],
]

NUMS = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二", "十三", "十四", "十五"]


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

    add_text(s, 12.05, 7.10, 1.0, 0.28, f"{i+1} / 15", 9,
             RGBColor(0xD9, 0xC8, 0x9A), bold=False, align=PP_ALIGN.RIGHT)

prs.save(PPT_OUT)
print("saved", PPT_OUT, len(prs.slides))
