"""以石大模板第 4 页为母版，生成 13 页。"""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).parent.parent
PPT_SRC = ROOT / "ppt" / "_石大空白_17页.pptx"
PPT_OUT = ROOT / "ppt" / "汇报_石大模板_17页.pptx"
NEW = ROOT / "output_real" / "three_parts" / "newdeck"

GOLD = RGBColor(0xC9, 0xA2, 0x27)
NAVY = RGBColor(0x0B, 0x1F, 0x3A)
CREAM = RGBColor(0xFB, 0xF7, 0xEA)
INK = RGBColor(0x1A, 0x1A, 0x1A)
RED = RGBColor(0xC0, 0x39, 0x2B)

prs = Presentation(PPT_SRC)

PAGES = [
    ("什么是地球重力场",
     "把逐点重力异常 Δg = g − γ 拼成整片场作为重建对象；"
     "因为 5–20 km 短波主要来自海底地形与断裂，导航与工程直接使用这一段。",
     NEW / "fig_new_concept_rich.png"),
    ("为什么要重建地球重力场",
     "按应用把可用波段拆开，重点补 5–20 km 短波；"
     "因为导航和地形/工程依赖这一段，大地测量与资源主要用长波。",
     NEW / "fig_new_why_matrix.png"),
    ("现有的 1′ 地球重力场产品",
     "用 SSV33.1 等 1′ 产品提供长波背景，再补 5–20 km 短波；"
     "因为 20 km 以下能量弱、且没有逐像元方差。",
     NEW / "fig_new_products_matrix.png"),
    ("从 1′ 到 15″ 的常用重建方法",
     "比较插值、频域融合和 LSC 三条路线，是为了看它们各自怎么把 1′ 与船测解算到 15″、短波从哪里来。",
     NEW / "fig_new_methods_3lanes.png"),
    ("这些方法存在的缺点",
     "检查三条路线的平滑、边界和协方差问题，是为了说明 5–20 km 不能靠它们稳定恢复。",
     NEW / "fig_new_cons_3methods.png"),
    ("标准 KF 可以解决上述问题",
     "用 KF 的观测更新、空间域递推和块增益对应三类问题，是为了避开插值平滑、FFT 边界和全局协方差求逆。",
     NEW / "fig_new_kf_mapping.png"),
    ("标准 KF 的流程",
     "把重力场写成状态、卫星和船测写成观测，用预测—增益—更新递推；"
     "这样做是为了让块观测、船测和先验按 R/Q 在同一套估计里分配权重。",
     NEW / "fig_new_stdkf_cycle.png"),
    ("标准 KF 存在的问题",
     "先检查标准 KF 直接套用的协方差和更新方式，是为了找出格点独立、块平均、条带和单点更新四个问题。",
     NEW / "fig_new_stdkf_problems_2x2.png"),
    ("相干保持 KF 的预测算子",
     "针对对角 P 把格点当成独立、单点更新只改一个像素，相干保持 KF 用结构张量相干 C 改写预测矩阵："
     "A=(1−wC²)Gσ+wC²I，令 x⁻=Ax；预测时邻域参与，船测修正沿构造扩散，脊线处不被抹平，平坦区正常平滑。",
     NEW / "fig_new_contribution_formula.png"),
    ("本项目 KF 流程",
     "卫星 1′、船测和地形先进入同一套 15″ 状态：块平均得到 1′ 观测，地形高频构造先验；"
     "迭代 6 轮并做 RTS 平滑。这样做是为了让块观测、先验和船测在同一递推中分配权重。",
     NEW / "fig_new_kf_flow_full.png"),
    ("KF 的迭代重建",
     "先以 SSV 与地形高频构造初始先验，再把 1′ 块观测和船测逐轮加入，6 轮后得到后验；"
     "先验只作初始场、后续由观测更新，是为了避免把先验当成观测重复加权。",
     NEW / "fig_new_iterations_mixed.png"),
    ("KF 的重建过程",
     "展示 C、wC²、预测修正和船测更新四个中间量，是为了看清相干保持 KF 改的是哪一步，而不只是看最终精度。",
     NEW / "fig_new_intermediates.png"),
    ("精度总结",
     "用整条航次留出比较双线性、频域融合、LSC 与相干保持 KF，是为了看相对常用方法的真实增益；"
     "四种场都以 SSV33.1 的 1′ 为输入。",
     NEW / "fig_new_accuracy_table.png"),
    ("三种常用场与相干保持 KF 的差异",
     "把双线性、频域融合、LSC 与相干保持 KF 放在同一日本区窗口：上排是场，下排是相干保持 KF 相对各场的差值；"
     "差值集中在构造与测线附近，说明相干保持 KF 补的是短波，不是整体平移。",
     NEW / "fig_new_methods_diff.png"),
    ("验证",
     "三个船测区按整条航次留出，训练时不使用该航次；"
     "下图取日本区一段连续剖面，比较船测、双线性、频域融合、LSC 与相干保持 KF。这样检验的是对未知测线的预测能力。",
     NEW / "fig_new_validation_profile.png"),
    ("断层",
     "把 SSV33.1（1′）与相干保持 KF 场放在同一套 5–20 km 带通、THG 和 60% 分位阈值下提取线性体，"
     "再与 GM9 对比；这样比较的是输入场，而不是换方法。",
     NEW / "fig_new_fault_metrics.png"),
    ("总结",
     "把 1′、船测和地形放进同一套 15″ KF 递推，并用结构张量相干改写预测矩阵；"
     "这样做是为了在保住长波的同时补出 5–20 km 短波，让同一套场同时服务精度与断裂结构。",
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
    ["初始先验", "1′ 块观测", "船测", "6 轮后验"],
    ["C 场", "wC²", "预测修正", "船测更新"],
    ["整条航次留出", "三种常用场", "相干保持 KF"],
    ["双线性", "频域融合", "LSC", "相干保持 KF−常用场"],
    ["整条航次留出", "连续剖面", "四种常用场", "误差"],
    ["SSV33.1", "相干保持 KF", "5–20 km", "THG"],
    ["15″ 场", "精度", "短波结构"],
]

NUMS = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二", "十三", "十四", "十五", "十六", "十七"]


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

    add_text(s, 12.05, 7.10, 1.0, 0.28, f"{i+1} / 17", 9,
             RGBColor(0xD9, 0xC8, 0x9A), bold=False, align=PP_ALIGN.RIGHT)

prs.save(PPT_OUT)
print("saved", PPT_OUT, len(prs.slides))
