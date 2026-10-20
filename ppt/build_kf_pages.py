"""KF 方法与结果 4 页：7.2/7.3 原理，8.5/8.6 结果。"""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).parent
FIGS = ROOT.parent / "output_real" / "three_parts"
OUT = ROOT / "KF方法与结果_五页.pptx"

NAVY = RGBColor(0x16, 0x36, 0x5C); MID = RGBColor(0x4A, 0x7A, 0xA7)
LIGHT = RGBColor(0xD8, 0xE0, 0xEC); GREY = RGBColor(0x5B, 0x65, 0x73)
PALE = RGBColor(0xEE, 0xF3, 0xFA); RED = RGBColor(0xC0, 0x39, 0x2B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "微软雅黑"

prs = Presentation()
prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def style(run, size, color, bold):
    run.font.size = Pt(size); run.font.color.rgb = color
    run.font.bold = bold; run.font.name = FONT


def rect(s, x, y, w, h, color):
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                            Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = color
    sh.line.fill.background(); sh.shadow.inherit = False


def textbox(s, x, y, w, h, paras, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
            spacing=None):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    first = True
    for para in paras:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = align
        if spacing:
            p.line_spacing = spacing
        if isinstance(para, tuple) and len(para) == 4 and isinstance(para[0], str):
            runs = [para]
        else:
            runs = para
        for txt, size, color, bold in runs:
            r = p.add_run(); style(r, size, color, bold); r.text = txt


def P(*runs):
    return list(runs)


def banner(s, code, main, section, sub):
    rect(s, 0, 0, 13.333, 0.68, NAVY)
    rect(s, 0, 0.68, 13.333, 0.042, RED)
    textbox(s, 0.42, 0.14, 9.8, 0.44, [(f"{code} ｜ {main}", 20, WHITE, True)])
    textbox(s, 10.1, 0.19, 2.8, 0.32,
            [("卡尔曼滤波重建地球重力场", 9.5, LIGHT, False)],
            align=PP_ALIGN.RIGHT)
    rect(s, 0.42, 0.84, 0.92, 0.30, NAVY)
    textbox(s, 0.42, 0.885, 0.92, 0.24, [(section, 10.5, WHITE, True)],
            align=PP_ALIGN.CENTER)
    textbox(s, 1.50, 0.815, 11.3, 0.62, sub, spacing=1.25)
    rect(s, 0.42, 1.48, 12.49, 0.010, LIGHT)


def new_slide(code, main, section, sub):
    s = prs.slides.add_slide(BLANK); banner(s, code, main, section, sub)
    return s


def picture(s, fname, x, y, max_w, max_h):
    p = FIGS / fname
    iw, ih = Image.open(p).size
    w = max_w; h = w * ih / iw
    if h > max_h:
        h = max_h; w = h * iw / ih
    s.shapes.add_picture(str(p), Inches(x), Inches(y), Inches(w), Inches(h))
    return w, h


def panel(s, x, y, w, h, bar=NAVY):
    rect(s, x, y, w, h, PALE); rect(s, x, y, 0.075, h, bar)


def bullets(s, x, y, w, items, size=11, gap=0.62):
    for i, it in enumerate(items):
        yy = y + i * gap
        rect(s, x, yy + 0.10, 0.075, 0.075, RED)
        textbox(s, x + 0.16, yy, w - 0.16, gap - 0.05,
                [(it, size, GREY, False)], spacing=1.3)


def strip(s, segs):
    rect(s, 0.42, 6.74, 12.49, 0.56, PALE)
    rect(s, 0.42, 6.74, 0.075, 0.56, RED)
    textbox(s, 0.78, 6.85, 11.95, 0.40,
            [[(t, 15, RED if c == "r" else NAVY, True) for t, c in segs]])


def N(t): return (t, "n")
def R(t): return (t, "r")


# ---------------------------------------------------------------- 1.3 三个应用
s = new_slide("1.3", "卡尔曼滤波支撑的三项应用", "背景", [
    P(("在 15″ 重力场基础上，用卡尔曼滤波同时完成", 13.5, GREY, False),
      ("重建、断裂识别与海底地形反演", 13.5, RED, True),
      ("；三者的状态与观测不同。", 13.5, GREY, False)),
])
picture(s, "fig_three_tasks.png", 0.42, 1.60, 12.49, 3.50)
for k, (t, body) in enumerate([
        ("应用一　重建", "状态：15″ 重力场；观测：卫星块均值、船载点；输出：15″ 场"),
        ("应用二　断裂", "状态：尺度分量；观测：同一重力场；输出：线性体"),
        ("应用三　反演", "状态：地形谱系数；观测：重力谱；输出：海底地形"),
]):
    x = 0.42 + k * 4.18
    panel(s, x, 5.20, 3.98, 1.28, [NAVY, MID, RED][k])
    textbox(s, x + 0.22, 5.30, 3.55, 0.28, [(t, 12, NAVY, True)])
    textbox(s, x + 0.22, 5.62, 3.55, 0.80,
            [(body, 9.5, GREY, False)], spacing=1.25)
strip(s, [N("卡尔曼滤波是方法，"), R("重建、断裂与反演"), N("是它的三个应用出口")])


# ---------------------------------------------------------------- 7.2 原理
s = new_slide("7.2", "KF 尺度分解：状态与观测", "方法", [
    P(("状态拆为 ", 13.5, GREY, False), ("x = x1 + x2 + x3", 13.5, RED, True),
      ("，三个分量用不同 σ 预测、共享同一套卫星与船载观测，增益按方差占比分配。", 13.5, GREY, False)),
])
picture(s, "fig_kf_scales_real.png", 0.42, 1.60, 12.49, 3.45)
for k, (t, body) in enumerate([
        ("状态", "x₁ 粗（σ=16 px）、x₂ 中（σ=4 px）、x₃ 细（σ=1 px）"),
        ("预测", "各分量高斯平滑，方差独立膨胀 Pᵢ←(1+0.1σ)Pᵢ+Qᵢ"),
        ("更新", "观测 z=H(x₁+x₂+x₃)，增益 Kᵢ=Pᵢ/ΣP+R"),
]):
    x = 0.42 + k * 4.18
    panel(s, x, 5.20, 3.98, 1.28, [NAVY, MID, RED][k])
    textbox(s, x + 0.22, 5.30, 3.55, 0.28, [(t, 12, NAVY, True)])
    textbox(s, x + 0.22, 5.62, 3.55, 0.80,
            [(body, 9.5, GREY, False)], spacing=1.25)
strip(s, [N("与 7.1 重建同属一套 KF："), R("状态分量"), N(" 代替单一状态，观测方程不变")])

# ---------------------------------------------------------------- 7.3 原理
s = new_slide("7.3", "KF 反演：谱域状态与增益", "方法", [
    P(("把地形谱系数作为状态、重力谱作为观测，逐波数做卡尔曼增益，", 13.5, GREY, False),
      ("无需矩阵求逆", 13.5, RED, True),
      ("，再叠加 Parker 非线性项。", 13.5, GREY, False)),
])
picture(s, "fig_kf_inversion_gain.png", 0.42, 1.60, 11.00, 3.50)
for k, (t, body) in enumerate([
        ("状态", "H(k)：地形谱系数；先验 P₀(k) 为幂律谱"),
        ("观测", "D(k)：重力谱；噪声 R 由重力精度给定"),
        ("增益", "K=P₀A/(A²P₀+R)，等价于阻尼低通"),
]):
    x = 0.42 + k * 4.18
    panel(s, x, 5.20, 3.98, 1.28, [NAVY, MID, RED][k])
    textbox(s, x + 0.22, 5.30, 3.55, 0.28, [(t, 12, NAVY, True)])
    textbox(s, x + 0.22, 5.62, 3.55, 0.80,
            [(body, 9.5, GREY, False)], spacing=1.25)
strip(s, [N("与 7.1 同一 KF 框架："), R("观测方程 A"), N(" 换成 Parker 正演，预测换成谱域先验")])

# ---------------------------------------------------------------- 8.5 结果
s = new_slide("8.5", "断裂识别结果：GM9 海域验证", "结果", [
    P(("留出法：22 条训练选参、22 条留出报告；缓冲 1–20 km，", 13.5, GREY, False),
      ("随机点作基线", 13.5, RED, True), ("。", 13.5, GREY, False)),
])
picture(s, "fig_fault_validation_curve.png", 0.42, 1.62, 6.90, 4.35)
panel(s, 7.55, 1.62, 5.35, 4.55)
textbox(s, 7.82, 1.72, 4.9, 0.30, [("验证设计", 12, NAVY, True)])
bullets(s, 7.82, 2.06, 4.95, [
    "留出法：训练段选阈值，留出段报结果",
    "缓冲半径扫描 1–20 km，随机点同口径作基线",
    "指标：覆盖率（断层被命中）与命中率（线性体靠断层）",
], size=9.2, gap=0.58)
textbox(s, 7.82, 3.95, 4.9, 0.30, [("结果（留出段）", 12, NAVY, True)])
bullets(s, 7.82, 4.29, 4.95, [
    "2 km：覆盖 22.2%（随机 3.4%）",
    "5 km：56.2%（随机 7.8%），命中 50.6%",
    "10 km：72.9%（随机 16.0%）",
], size=9.2, gap=0.56)
textbox(s, 7.82, 5.95, 4.85, 0.30,
        [("频带（5–20 km）与海域断层编目是主要控制因素。", 8.8, GREY, False)])
strip(s, [N("结论："), R("5 km 缓冲覆盖 56.2%"), N("，留出断层段验证")])

# ---------------------------------------------------------------- 8.6 结果
s = new_slide("8.6", "反演结果：KF 与 Parker 对照", "结果", [
    P(("分频评估 + Parker 同参数对照 + ", 13.5, GREY, False),
      ("GMRT 独立产品交叉验证", 13.5, RED, True),
      ("，三条线同时报告。", 13.5, GREY, False)),
])
picture(s, "fig_inversion_validation.png", 0.42, 1.62, 6.90, 3.20)
textbox(s, 0.50, 5.05, 6.80, 1.35, [
    [("频段", 9, NAVY, True), ("        RMSE", 9, NAVY, True),
     ("     corr", 9, NAVY, True)],
    [("20–50 km", 9, GREY, False), ("      244 m", 9, GREY, False),
     ("      0.44", 9, GREY, False)],
    [("50–100 km", 9, GREY, False), ("     288 m", 9, GREY, False),
     ("      0.54", 9, GREY, False)],
    [("100–200 km", 9, GREY, False), ("    355 m", 9, GREY, False),
     ("      0.75", 9, GREY, False)],
], spacing=1.25)
panel(s, 7.55, 1.62, 5.35, 4.55)
textbox(s, 7.82, 1.72, 4.9, 0.30, [("验证设计", 12, NAVY, True)])
bullets(s, 7.82, 2.06, 4.95, [
    "带通 20–200 km，分频 20–50 / 50–100 / 100–200 km",
    "Parker 用同一频带、同一密度与低通作对照",
    "GMRT 为独立地形产品，先与 GEBCO 配准（corr 0.997）",
], size=9.2, gap=0.58)
textbox(s, 7.82, 3.95, 4.9, 0.30, [("结果", 12, NAVY, True)])
bullets(s, 7.82, 4.29, 4.95, [
    "分频 RMSE：244 / 288 / 355 m",
    "全带：KF 650.8 m，Parker 650.7 m，corr 0.61",
    "GMRT 交叉验证 648 m / corr 0.608",
], size=9.2, gap=0.56)
textbox(s, 7.82, 5.95, 4.85, 0.30,
        [("全带误差由 100–200 km 深部场贡献；常密度与低通是主要偏差来源。", 8.8, GREY, False)])
strip(s, [N("结论：KF 反演与 Parker 等价，"), R("20–50 km 波段 244 m")])

prs.save(str(OUT))
print("saved", OUT, "pages", len(prs.slides))


