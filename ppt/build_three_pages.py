"""生成“三部分研究内容”3 页 PPT（沿用原稿的深蓝页眉 + 药丸 + 结论带样式）。"""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).parent
FIGS = ROOT.parent / "output_real" / "three_parts"
OUT = ROOT / "三部分研究内容_三页.pptx"

NAVY = RGBColor(0x16, 0x36, 0x5C)
NAVY_D = RGBColor(0x0E, 0x27, 0x42)
MID = RGBColor(0x4A, 0x7A, 0xA7)
LIGHT = RGBColor(0xD8, 0xE0, 0xEC)
GREY = RGBColor(0x5B, 0x65, 0x73)
PALE = RGBColor(0xEE, 0xF3, 0xFA)
RED = RGBColor(0xC0, 0x39, 0x2B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLUE = RGBColor(0x1F, 0x4E, 0x79)
FONT = "微软雅黑"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def style(run, size, color, bold):
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.name = FONT


def rect(s, x, y, w, h, color):
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                            Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = color
    sh.line.fill.background(); sh.shadow.inherit = False
    return sh


def textbox(s, x, y, w, h, paras, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
            spacing=None):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    first = True
    for para in paras:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = align
        if spacing:
            p.line_spacing = spacing
        runs = para if isinstance(para, list) else [para]
        for txt, size, color, bold in runs:
            r = p.add_run(); style(r, size, color, bold); r.text = txt
    return tb


def para(*runs):
    return list(runs)


def N(t):
    return (t, "n")


def R(t):
    return (t, "r")


def strip(s, segs):
    if isinstance(segs, tuple):
        segs = [segs]
    rect(s, 0.42, 6.74, 12.49, 0.56, PALE)
    rect(s, 0.42, 6.74, 0.075, 0.56, RED)
    textbox(s, 0.78, 6.85, 11.95, 0.40,
            [[(t, 15, RED if c == "r" else NAVY, True) for t, c in segs]])


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
    s = prs.slides.add_slide(BLANK)
    banner(s, code, main, section, sub)
    return s


def picture(s, fname, top, max_w, max_h, left=None):
    p = FIGS / fname
    iw, ih = Image.open(p).size
    w = max_w; h = w * ih / iw
    if h > max_h:
        h = max_h; w = h * iw / ih
    x = (13.333 - w) / 2 if left is None else left
    s.shapes.add_picture(str(p), Inches(x), Inches(top), Inches(w), Inches(h))


def bullets(s, x, y, w, items, size=12, gap=0.62):
    for i, item in enumerate(items):
        yy = y + i * gap
        rect(s, x, yy + 0.10, 0.075, 0.075, RED)
        textbox(s, x + 0.16, yy, w - 0.16, gap - 0.05,
                [(item, size, GREY, False)], spacing=1.3)


def panel(s, x, y, w, h, color=PALE, bar=NAVY):
    rect(s, x, y, w, h, color)
    rect(s, x, y, 0.075, h, bar)


# ================================================================ 页 2
s = new_slide("7.2", "KF 尺度分解与断裂识别", "方法", [
    para(("KF 场 + ", 13.5, GREY, False),
         ("5–20 km 尺度带", 13.5, RED, True),
         ("提线性体，在 GM9 海域窗口按 2/5/10 km 缓冲、留出断层段验证。", 13.5, GREY, False)),
])
picture(s, "fig_gm9_faults.png", 1.62, 7.10, 4.40, left=0.42)
panel(s, 7.75, 1.62, 5.15, 4.55)
textbox(s, 8.02, 1.74, 4.7, 0.32, [("GM9 海域断层验证（留出段）", 12.5, NAVY, True)])
bullets(s, 8.02, 2.14, 4.75, [
    "2 km：断层覆盖 22.2%（随机 3.4%）",
    "5 km：56.2%（随机 7.8%），命中 50.6%",
    "10 km：72.9%（随机 16.0%）",
    "与文献 2–5 km 缓冲的 30–60% 同量级",
], size=9.8, gap=0.70)
textbox(s, 8.02, 5.10, 4.70, 0.95,
        [("频带（5–20 km）与海域断层编目是主要控制因素；区域级 GEM 基准仅 0.6–3.4%。",
          9.2, GREY, False)], spacing=1.3)

strip(s, [N("KF 尺度分量为断裂识别提供可解释的线性体，空间吻合度以"), R("海域断层编目"), N("检验")])

# ================================================================ 页 3
s = new_slide("7.3", "KF 反演海底地形", "方法", [
    para(("用", 13.5, GREY, False),
         ("谱域卡尔曼滤波反演", 13.5, RED, True),
         ("（状态=地形谱系数，先验谱+观测噪声）由 15″ 重力场反演海底地形，与 Parker 等价，用 GEBCO 带通检验。", 13.5, GREY, False)),
])
picture(s, "fig_inversion_eval.png", 1.62, 7.10, 4.10, left=0.42)
panel(s, 7.75, 1.62, 5.15, 4.55)
textbox(s, 8.02, 1.74, 4.7, 0.32, [("精度与量级（KF 反演）", 12.5, NAVY, True)])
bullets(s, 8.02, 2.14, 4.75, [
    "20–50 km 波段 RMSE 244 m，落在文献 100–300 m 区间",
    "20–100 km 405 m；全带 20–200 km 651 m（长波拉高）",
    "KF 与 Parker 等价：650.8 m vs 650.7 m，corr 0.61",
    "残差 24.85 → 0.92 mGal（8 轮）",
], size=9.8, gap=0.70)
textbox(s, 8.02, 5.10, 4.70, 1.00,
        [("对照：文献 100–300 m 多对应 20–50 km 波段，本区 244 m 已落在区间内；全带 651 m 偏高但同量级，主因 100–200 km 深部场与常密度。",
          9.2, GREY, False)], spacing=1.3)

strip(s, [N("谱域 KF 反演与 Parker 阻尼反演等价，精度由"), R("频带与密度模型"), N("决定")])

prs.save(str(OUT))
print("saved", OUT, "pages", len(prs.slides))
