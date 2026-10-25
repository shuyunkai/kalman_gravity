"""生成一页汇报模板：上方深色文字重点，下方图片。"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).parent.parent
FIG = ROOT / "output_real" / "three_parts" / "fig_fault_method_4.png"
OUT = Path(__file__).parent / "汇报模板_一页.pptx"

NAVY = RGBColor(0x0B, 0x25, 0x45)
NAVY2 = RGBColor(0x14, 0x3A, 0x63)
RED = RGBColor(0xC0, 0x39, 0x2B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREY = RGBColor(0x5B, 0x65, 0x73)
LIGHT = RGBColor(0xEE, 0xF3, 0xF9)
PALE = RGBColor(0xD6, 0xE2, 0xF0)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
s = prs.slides.add_slide(prs.slide_layouts[6])


def rect(x, y, w, h, color):
    from pptx.enum.shapes import MSO_SHAPE
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                            Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    return sh


def text(x, y, w, h, runs, size=12, color=GREY, bold=False,
         align=PP_ALIGN.LEFT, spacing=1.0, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0; tf.margin_right = 0
    tf.margin_top = 0; tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.alignment = align
    p.line_spacing = spacing
    for txt, sz, col, bd in runs:
        r = p.add_run(); r.text = txt
        r.font.size = Pt(sz); r.font.color.rgb = col
        r.font.bold = bd; r.font.name = "Microsoft YaHei"
    return tb


# 上方深色区
rect(0, 0, 13.333, 2.45, NAVY)
rect(0, 2.45, 13.333, 0.06, RED)

text(0.55, 0.28, 8.0, 0.35,
     [("二、研究问题与学术思路", 15, PALE, False)])
text(0.55, 0.72, 11.8, 0.65,
     [("空间相关自适应卡尔曼滤波", 31, WHITE, True)])
text(0.55, 1.45, 11.8, 0.38,
     [("多源观测融合的 15″ 地球重力场重建", 16, PALE, False)])

# 三个标签
labels = [("卫星 1′ + 船测", 0.55), ("协方差 P∘L", 3.55), ("R_sat 自适应", 6.55)]
for txt, x in labels:
    from pptx.enum.shapes import MSO_SHAPE
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.92),
                            Inches(2.65), Inches(0.38))
    sh.fill.background()
    sh.line.color.rgb = RGBColor(0x5B, 0x7F, 0xA8); sh.line.width = Pt(0.8)
    tf = sh.text_frame; tf.margin_top = 0; tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = txt
    r.font.size = Pt(10.5); r.font.color.rgb = PALE
    r.font.name = "Microsoft YaHei"

# 核心概念
from pptx.enum.shapes import MSO_SHAPE
pill = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.55), Inches(2.72),
                          Inches(1.28), Inches(0.38))
pill.fill.solid(); pill.fill.fore_color.rgb = NAVY
pill.line.fill.background()
tf = pill.text_frame; tf.margin_top = 0; tf.margin_bottom = 0
p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
r = p.add_run(); r.text = "核心概念"
r.font.size = Pt(11); r.font.color.rgb = WHITE; r.font.bold = True
r.font.name = "Microsoft YaHei"

text(2.0, 2.69, 10.7, 0.5,
     [("用空间相关的协方差修正增益，让船测修正按相关长度扩散，而不是更新后人为抹平。",
       12.5, GREY, False)])

# 三条要点
bullets = [
    ("观测", "卫星 1′ 块平均 + 船测点更新，统一到 15″ 状态"),
    ("协方差", "P∘L 引入空间相关，相关长度 ℓ≈13 km"),
    ("噪声", "新息在线估计 R_sat；R_ship 固定 10 mGal²"),
]
y = 3.28
for head, body in bullets:
    text(0.65, y, 1.1, 0.3, [(head, 11.5, RED, True)])
    text(1.75, y - 0.01, 10.8, 0.3, [(body, 11.5, GREY, False)])
    y += 0.34

# 图片
pic = s.shapes.add_picture(str(FIG), Inches(1.75), Inches(4.20),
                           width=Inches(9.8))
rect(0.55, 6.85, 12.2, 0.42, RED)
text(0.75, 6.90, 11.8, 0.32,
     [("结论：15″ 场 → 5–20 km 带通 + THG → 断裂线性体，5 km 覆盖 61.6%",
       11.5, WHITE, True)])

text(12.2, 7.32, 0.7, 0.18, [("1 / 1", 8.5, GREY, False)])

prs.save(OUT)
print("saved", OUT)
