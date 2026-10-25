"""深色区：标题 24pt、正文 17pt 两种字号；下方只放图。"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).parent.parent
FIG = ROOT / "output_real" / "three_parts" / "fig_paper_method_flow.png"
OUT = Path(__file__).parent / "汇报模板_大字版.pptx"

NAVY = RGBColor(0x0B, 0x25, 0x45)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PALE = RGBColor(0xC9, 0xD9, 0xEA)
RED = RGBColor(0xC0, 0x39, 0x2B)

TITLE = 24
BODY = 17

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


def text(x, y, w, h, txt, size, color, bold=False, spacing=1.25,
         align=PP_ALIGN.LEFT):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = 0; tf.margin_right = 0
    tf.margin_top = 0; tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.TOP
    p = tf.paragraphs[0]; p.alignment = align
    p.line_spacing = spacing
    r = p.add_run(); r.text = txt
    r.font.size = Pt(size); r.font.color.rgb = color
    r.font.bold = bold; r.font.name = "Microsoft YaHei"


# 深色区
rect(0, 0, 13.333, 3.78, NAVY)

# 标题（两种字号中的“标题”）
text(0.7, 0.30, 11.9, 0.45, "二、研究问题与学术思路", TITLE, PALE, True)
text(0.7, 0.78, 11.9, 0.55, "2.3 空间相关自适应卡尔曼滤波",
     TITLE, WHITE, True)

# 正文
text(0.7, 1.48, 11.9, 1.0,
     "针对卫星 1′ 足迹与船测稀疏，把 15″ 重力场作为状态、"
     "卫星块平均与船测点作为观测；用 P∘L 修正协方差引入空间相关，"
     "用新息自适应估计 R_sat。",
     BODY, PALE, False, spacing=1.45)

# 结论：加粗
rect(0.7, 2.72, 0.07, 0.55, RED)
text(0.95, 2.72, 11.6, 0.6,
     "结论：同一套 15″ 场进入断裂识别，5 km 覆盖 61.6%。",
     BODY, WHITE, True, spacing=1.2)

# 页码：仍用正文字号
text(11.9, 3.40, 0.8, 0.3, "3 / 24", BODY, PALE, align=PP_ALIGN.RIGHT)

# 下方只放图
s.shapes.add_picture(str(FIG), Inches(1.82), Inches(3.84),
                     width=Inches(9.6))

prs.save(OUT)
print("saved", OUT)
