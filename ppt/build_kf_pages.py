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
OUT = ROOT / "KF方法与结果_六页.pptx"

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
s = new_slide("7.2", "断裂提取：带通 + THG", "方法", [
    P(("在 KF 15″ 场上做 ", 13.5, GREY, False), ("5–20 km 带通", 13.5, RED, True),
      ("，取 THG 脊线，细化后按面积/伸长率过滤，得到线性体。", 13.5, GREY, False)),
])
picture(s, "fig_fault_method_4.png", 0.42, 1.55, 12.49, 3.55)
for k, (t, body) in enumerate([
        ("带通", "保留 5–20 km 构造尺度，压制长波与高频噪声"),
        ("THG", "水平总梯度脊线，对断层边界敏感"),
        ("过滤", "Zhang–Suen 细化；面积≥20 px、伸长率≥1.8"),
]):
    x = 0.42 + k * 4.18
    panel(s, x, 5.20, 3.98, 1.28, [NAVY, MID, RED][k])
    textbox(s, x + 0.22, 5.30, 3.55, 0.28, [(t, 12, NAVY, True)])
    textbox(s, x + 0.22, 5.62, 3.55, 0.80,
            [(body, 9.5, GREY, False)], spacing=1.25)
strip(s, [N("常规图像算法，输入是"), R("KF 重建场")])

# ---------------------------------------------------------------- 7.3 原理
s = new_slide("7.3", "Parker–Oldenburg 反演", "方法", [
    P(("重力谱经 ", 13.5, GREY, False), ("Parker 正演迭代", 13.5, RED, True),
      ("反解地形；KF 谱域增益与之一致，仅作对照。", 13.5, GREY, False)),
])
picture(s, "fig_parker_principle.png", 0.42, 1.55, 12.49, 3.55)
for k, (t, body) in enumerate([
        ("正演", "Δg(k) 由地形谱 h 的幂次项求和，含 e^(−kz₀)"),
        ("反演", "逐波数反解，迭代 8 次，低通 25 km，松弛 0.6"),
        ("对照", "KF 谱域增益 K=P₀A/(A²P₀+R)，结果与 Parker 一致"),
]):
    x = 0.42 + k * 4.18
    panel(s, x, 5.20, 3.98, 1.28, [NAVY, MID, RED][k])
    textbox(s, x + 0.22, 5.30, 3.55, 0.28, [(t, 12, NAVY, True)])
    textbox(s, x + 0.22, 5.62, 3.55, 0.80,
            [(body, 9.5, GREY, False)], spacing=1.25)
strip(s, [N("标准 Parker–Oldenburg 反演，"), R("KF 谱域仅作对照")])

# ---------------------------------------------------------------- 6.2 方法总览
s = new_slide("6.2", "方法总览：KF 驱动的三项任务", "流程", [
    P(("同一套 15″ KF 场分别进入", 13.5, GREY, False),
      ("5–20 km 带通与 Parker 反演", 13.5, RED, True),
      ("，再由外部船测与断层编目验证。", 13.5, GREY, False)),
])
picture(s, "fig_paper_method_flow.png", 0.87, 1.52, 11.60, 4.32)
strip(s, [N("KF 负责重建，断裂与反演共享同一套 "), R("15″ 场")])

# ---------------------------------------------------------------- 8.5 结果
s = new_slide("8.5", "断裂识别结果：GM9 海域验证", "结果", [
    P(("以", 13.5, GREY, False), ("留出断层段", 13.5, RED, True),
      ("检验 5–20 km 线性体与 GM9 海域断层的空间吻合度。", 13.5, GREY, False)),
])
picture(s, "fig_paper_fault_validation.png", 0.42, 1.58, 12.49, 3.55)
for k, (t, body) in enumerate([
        ("验证设计", "22 条训练选参、22 条留出报告；缓冲 1–20 km；覆盖率随缓冲半径变化"),
        ("结果", "5 km 61.6%｜10 km 74.1%｜20 km 74.4%"),
        ("说明", "61.6% 的留出断层段在 5 km 内被线性体覆盖；走向与区域构造一致"),
]):
    x = 0.42 + k * 4.18
    panel(s, x, 5.25, 3.98, 1.28, [NAVY, MID, RED][k])
    textbox(s, x + 0.22, 5.35, 3.55, 0.28, [(t, 12, NAVY, True)])
    textbox(s, x + 0.22, 5.67, 3.55, 0.80,
            [(body, 9.0, GREY, False)], spacing=1.2)
strip(s, [N("结论：5 km 缓冲下"), R("61.6%"), N(" 的留出断层段被线性体覆盖")])

# ---------------------------------------------------------------- 8.6 结果
s = new_slide("8.6", "反演结果：KF 与 Parker 对照", "结果", [
    P(("从", 13.5, GREY, False), ("分频精度、反演—真值一致性、独立地形产品", 13.5, RED, True),
      ("三方面检验反演结果。", 13.5, GREY, False)),
])
picture(s, "fig_paper_inversion_validation.png", 0.42, 1.58, 12.49, 3.55)
for k, (t, body) in enumerate([
        ("验证设计", "KF 场输入标准 Parker 反演；20–200 km 三分频；KF 谱域作对照"),
        ("结果", "20–50 km 112 m｜50–100 km 190 m｜100–200 km 363 m；20–200 km 372 m"),
        ("交叉验证", "内区去 40 px 边缘，corr 0.80；KF 谱域同解 388 m / 0.79"),
]):
    x = 0.42 + k * 4.18
    panel(s, x, 5.25, 3.98, 1.28, [NAVY, MID, RED][k])
    textbox(s, x + 0.22, 5.35, 3.55, 0.28, [(t, 12, NAVY, True)])
    textbox(s, x + 0.22, 5.67, 3.55, 0.80,
            [(body, 9.0, GREY, False)], spacing=1.2)
strip(s, [N("Parker：20–50 km"), R("112 m"), N("，20–200 km 372 m，corr 0.80")])

# ---------------------------------------------------------------- 8.7 跨区域
s = new_slide("8.7", "跨区域断裂验证", "结果", [
    P(("三个海域窗口、同一套 KF + 5–20 km 流程与匹配口径，比较", 13.5, GREY, False),
      ("样本量与信号强弱", 13.5, RED, True), ("。", 13.5, GREY, False)),
])
rows = [
    ("区域", "断层(条)", "海域(条)", "5 km 覆盖"),
    ("GM9 海域", "44", "44", "61.6%"),
    ("爱琴海", "161", "99", "24.3%"),
    ("加州近海", "233", "78", "0%"),
]
for i, r in enumerate(rows):
    y = 1.70 + i * 0.52
    if i == 0:
        rect(s, 0.42, y - 0.06, 7.6, 0.46, NAVY)
    elif i % 2 == 0:
        rect(s, 0.42, y - 0.06, 7.6, 0.46, PALE)
    for j, txt in enumerate(r):
        textbox(s, 0.55 + j * 1.85, y, 1.70, 0.34,
                [(txt, 10.5, WHITE if i == 0 else GREY, i == 0)], spacing=1.0)
textbox(s, 0.42, 3.95, 7.6, 0.32,
        [("对照：GM9 为 22 条留出断层段；爱琴海 10,151 个海域断层点；加州 2,526 个。", 8.8, GREY, False)])
textbox(s, 0.42, 4.45, 7.6, 1.10, [
    [("15″ 是网格分辨率，不是重力信息分辨率：", 9.2, RED, True),
     ("KF 场与 SSV 的差异在 2–5 km 仅 0.14 mGal，", 9.2, GREY, False)],
    [("5–20 km 与地形先验相关 0.83–0.96；", 9.2, GREY, False),
     ("短波主要来自地形先验，不是独立重力观测。", 9.2, GREY, False)],
], spacing=1.3)
panel(s, 8.30, 1.64, 4.60, 4.55)
textbox(s, 8.56, 1.74, 4.1, 0.30, [("区域对比结论", 12, NAVY, True)])
bullets(s, 8.56, 2.10, 4.15, [
    "GM9：信号最强，但仅 22 条留出，样本小",
    "爱琴海：样本大，但 5 km 覆盖仅 24.3%",
    "加州：线性体与断层空间不重叠，0%",
    "结论对样本与频带敏感，需分区报告",
], size=9.0, gap=0.92)
strip(s, [N("结论：方法可跨区，信号强弱取决于"), R("海域断层编目与频带匹配")])

# ---------------------------------------------------------------- 8.8 消融
s = new_slide("8.8", "消融对比：KF 场 vs SSV 场", "结果", [
    P(("同一套带通/THG 与 Parker，只换输入场：", 13.5, GREY, False),
      ("KF 15″ 场 vs SSV 直接插值", 13.5, RED, True), ("。", 13.5, GREY, False)),
])
rows = [
    ("指标", "KF 15″ 场", "SSV 直接插值"),
    ("5 km 断层覆盖", "61.6%", "10.1%"),
    ("10 km 断层覆盖", "74.1%", "36.2%"),
    ("20 km 断层覆盖", "74.4%", "95.3%"),
    ("10–50 km RMSE", "124 m", "131 m"),
    ("10–50 km corr", "0.82", "0.77"),
    ("20–200 km RMSE", "372 m", "376 m"),
]
for i, r in enumerate(rows):
    y = 1.68 + i * 0.50
    if i == 0:
        rect(s, 0.42, y - 0.05, 8.2, 0.44, NAVY)
    elif i % 2 == 0:
        rect(s, 0.42, y - 0.05, 8.2, 0.44, PALE)
    for j, txt in enumerate(r):
        textbox(s, 0.55 + j * 2.75, y, 2.6, 0.32,
                [(txt, 10.0, WHITE if i == 0 else GREY, i == 0)], spacing=1.0)
panel(s, 8.85, 1.64, 4.05, 4.45)
textbox(s, 9.10, 1.74, 3.6, 0.30, [("结论", 12, NAVY, True)])
bullets(s, 9.10, 2.10, 3.65, [
    "5–10 km：KF 明显优于 SSV",
    "20 km 以上：SSV 覆盖率反超",
    "SSV 线稀疏，靠大缓冲凑覆盖",
    "反演 5–50 km：KF 更好",
    "5–20 km 含地形先验，需注明",
], size=8.6, gap=0.78)
strip(s, [N("KF 优势在"), R("5–10 km 空间精度"), N("；远距离覆盖率不占优")])

prs.save(str(OUT))
print("saved", OUT, "pages", len(prs.slides))




