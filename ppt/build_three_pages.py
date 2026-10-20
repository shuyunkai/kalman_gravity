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


# ================================================================ 页 1
s = new_slide("10.1", "研究内容一：15″ 重力场重建", "内容一", [
    para(("把卫星测高（1′）、船载重力（10 s 点）、海底地形（15″）写进", 13.5, GREY, False),
         ("同一套卡尔曼滤波方程", 13.5, RED, True),
         ("，重建 15″ 海洋重力场。", 13.5, GREY, False)),
])
panel(s, 0.42, 1.62, 6.05, 2.55)
textbox(s, 0.75, 1.72, 5.4, 0.32, [("方法要点", 14, NAVY, True)])
bullets(s, 0.75, 2.12, 5.55, [
    "状态：15″ 重力场，1.84×10⁷ 维；卫星观测 1.15×10⁶ 个",
    "先验：SSV 块填充 + 0.5×地形布格高频；6 轮 KF + RTS",
    "参数：α=0.5、σ=4、Q=4、R_sat=9、R_ship=1（三区冻结）",
    "验证：三个互不重叠区、整条航次留出、双口径评估",
], size=10.5, gap=0.50)

for k, (val, lab) in enumerate([("4.173", "综合验证区"), ("4.748", "南海-琉球"),
                                ("1.455", "菲律宾海中部")]):
    y = 1.62 + k * 0.88
    panel(s, 6.75, y, 6.15, 0.74, PALE, [NAVY, MID, RED][k])
    textbox(s, 7.00, y + 0.10, 1.35, 0.40, [(val, 17, NAVY, True)])
    textbox(s, 8.40, y + 0.14, 4.30, 0.34,
            [(f"{lab}　KF 15″ RMSE（mGal）", 10.5, GREY, False)])

panel(s, 6.75, 4.35, 6.15, 1.95, PALE, RED)
textbox(s, 7.00, 4.45, 5.7, 0.32, [("相对基线", 13, NAVY, True)])
bullets(s, 7.00, 4.82, 5.75, [
    "相对 SSV33.1：改善 9.4%–10.7%",
    "相对频域融合：改善 6.9%–8.8%",
    "三种方法同一留出航次、同一评估函数",
], size=10, gap=0.46)

strip(s, [N("这一段解决"), R("“精度”"), N("：同源数据、同口径，可复现")])

# ================================================================ 页 2
s = new_slide("10.2", "研究内容二：尺度分解与断裂识别", "内容二", [
    para(("把 KF 状态拆成", 13.5, GREY, False),
         ("粗 / 中 / 细三个分量", 13.5, RED, True),
         ("（方差按比例分配），用 KF 场做尺度分解，再提线性体并与 GEM 活动断层对比。", 13.5, GREY, False)),
])
picture(s, "fig_fault_kf_map.png", 1.62, 7.10, 4.40, left=0.42)
panel(s, 7.75, 1.62, 5.15, 4.55)
textbox(s, 8.02, 1.74, 4.7, 0.32, [("多尺度 KF vs GEM 活动断层", 12.5, NAVY, True)])
bullets(s, 8.02, 2.14, 4.75, [
    "三分量 KF：x1(粗)+x2(中)+x3(细)，共享观测",
    "20 km 内覆盖 40.7%（随机 2.3%）→ 约 17 倍",
    "50 km 内 52.6%（随机 6.8%），精度 30.3%",
    "对照：带通基线 20 km 仅 14.9%（随机 6.1%）",
], size=9.8, gap=0.70)
textbox(s, 8.02, 5.10, 4.70, 0.95,
        [("解读：KF 多尺度分解比固定带通明显提升；x2 单分量仍不稳定，σ/Q 需再标定。",
          9.2, GREY, False)], spacing=1.3)

strip(s, [N("这一段提供"), R("“构造解释”"), N("的入口：从 KF 尺度分量到线性体，再到公开断层检验")])

# ================================================================ 页 3
s = new_slide("10.3", "研究内容三：重力反演海底地形", "内容三", [
    para(("用 Parker–Oldenburg 方法由 15″ 重力场反演海底地形，", 13.5, GREY, False),
         ("常密度 + 频域阻尼迭代", 13.5, RED, True),
         ("，并用 GEBCO 带通地形检验。", 13.5, GREY, False)),
])
picture(s, "fig_inversion_eval.png", 1.62, 7.10, 4.10, left=0.42)
panel(s, 7.75, 1.62, 5.15, 4.55)
textbox(s, 8.02, 1.74, 4.7, 0.32, [("精度与量级", 12.5, NAVY, True)])
bullets(s, 8.02, 2.14, 4.75, [
    "20–50 km 波段 RMSE 244 m，落在文献 100–300 m 区间",
    "20–100 km 405 m；全带 20–200 km 651 m（长波拉高）",
    "振幅斜率 1.34 → 标定后 638 m；corr 0.61",
    "残差 24.85 → 0.88 mGal（8 轮）",
], size=9.8, gap=0.70)
textbox(s, 8.02, 5.10, 4.70, 1.00,
        [("对照：文献 100–300 m 多对应 20–50 km 波段，本区 244 m 已落在区间内；全带 651 m 偏高但同量级，主因 100–200 km 深部场与常密度。",
          9.2, GREY, False)], spacing=1.3)

strip(s, [N("这一段完成"), R("“应用闭环”"), N("：重力场 → 海底地形 → 可检验的精度")])

prs.save(str(OUT))
print("saved", OUT, "pages", len(prs.slides))
