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
OUT = ROOT / "KF方法与结果_四页.pptx"

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


# ---------------------------------------------------------------- 7.2 原理
s = new_slide("7.2", "KF 尺度分解：状态与观测", "方法", [
    P(("状态拆为 ", 13.5, GREY, False), ("x = x1 + x2 + x3", 13.5, RED, True),
      ("，三个分量用不同 σ 预测、共享同一套卫星与船载观测，增益按方差占比分配。", 13.5, GREY, False)),
])
picture(s, "fig_kf_scales_schematic.png", 0.42, 1.60, 12.49, 3.45)
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
    P(("KF 场 + 5–20 km 尺度带提线性体，在 GM9 海域窗口按 2/5/10 km 缓冲、", 13.5, GREY, False),
      ("留出断层段", 13.5, RED, True), ("评估。", 13.5, GREY, False)),
])
picture(s, "fig_gm9_faults.png", 0.42, 1.62, 7.00, 4.40)
panel(s, 7.75, 1.62, 5.15, 4.55)
textbox(s, 8.02, 1.74, 4.7, 0.32, [("验证口径与结果", 12.5, NAVY, True)])
bullets(s, 8.02, 2.14, 4.75, [
    "训练段 22 条选参，留出段 22 条报告",
    "2 km：覆盖 22.2%（随机 3.4%）",
    "5 km：56.2%（随机 7.8%），命中 50.6%",
    "10 km：72.9%（随机 16.0%）",
], size=9.8, gap=0.70)
textbox(s, 8.02, 5.10, 4.70, 0.95,
        [("与文献 2–5 km 缓冲的 30–60% 同量级；频带与海域断层编目是主要控制因素。",
          9.2, GREY, False)], spacing=1.3)
strip(s, [N("结论："), R("5 km 缓冲 56.2%"), N("，与公开研究同量级")])

# ---------------------------------------------------------------- 8.6 结果
s = new_slide("8.6", "反演结果：KF 与 Parker 对照", "结果", [
    P(("谱域 KF 反演与 Parker 阻尼反演并行计算，", 13.5, GREY, False),
      ("结果等价", 13.5, RED, True),
      ("；并用 GMRT 独立地形产品交叉验证。", 13.5, GREY, False)),
])
picture(s, "fig_inversion_eval.png", 0.42, 1.62, 7.00, 4.10)
panel(s, 7.75, 1.62, 5.15, 4.55)
textbox(s, 8.02, 1.74, 4.7, 0.32, [("精度与交叉验证", 12.5, NAVY, True)])
bullets(s, 8.02, 2.14, 4.75, [
    "KF 650.8 m / corr 0.61；Parker 650.7 m",
    "20–50 km 波段 244 m，落在文献 100–300 m",
    "GMRT 交叉验证 648 m / corr 0.608",
    "残差 24.85 → 0.92 mGal（8 轮）",
], size=9.8, gap=0.70)
textbox(s, 8.02, 5.10, 4.70, 0.95,
        [("全带 651 m 由 100–200 km 深部场贡献；常密度与低通是主要偏差来源。",
          9.2, GREY, False)], spacing=1.3)
strip(s, [N("结论：KF 反演与 Parker 等价，"), R("20–50 km 244 m"), N(" 与文献同量级")])

prs.save(str(OUT))
print("saved", OUT, "pages", len(prs.slides))
