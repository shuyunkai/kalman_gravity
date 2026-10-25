"""深色区承载全部文字，下方只放图。"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).parent.parent
FIG = ROOT / "output_real" / "three_parts" / "fig_paper_method_flow.png"
OUT = Path(__file__).parent / "汇报模板_文字在上.pptx"

NAVY = RGBColor(0x0B, 0x25, 0x45)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PALE = RGBColor(0xC9, 0xD9, 0xEA)
RED = RGBColor(0xC0, 0x39, 0x2B)

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


def text(x, y, w, h, txt, size, color, bold=False, spacing=1.3,
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


# 深色区：所有文字
rect(0, 0, 13.333, 3.62, NAVY)

text(0.7, 0.28, 9.0, 0.3, "二、研究问题与学术思路", 13, PALE)
text(10.6, 0.28, 2.1, 0.3, "2026.10　｜　3 / 24", 10.5, PALE,
     align=PP_ALIGN.RIGHT)

text(0.7, 0.66, 11.9, 0.55, "2.3 空间相关自适应卡尔曼滤波",
     26, WHITE, True)

text(0.7, 1.42, 11.9, 0.95,
     "针对卫星 1′ 足迹与船测稀疏，把 15″ 重力场作为状态、"
     "卫星块平均与船测点作为观测；用 P∘L 修正协方差引入空间相关，"
     "用新息自适应估计 R_sat，重建高分辨率重力场。",
     12.5, PALE, False, spacing=1.5)

rect(0.7, 2.62, 0.06, 0.55, RED)
text(0.92, 2.64, 11.6, 0.55,
     "结论：同一套 15″ 场进入断裂识别，5 km 覆盖 61.6%。",
     13, WHITE, True, spacing=1.2)

# 下方只放图
s.shapes.add_picture(str(FIG), Inches(1.72), Inches(3.76),
                     width=Inches(9.9))

prs.save(OUT)
print("saved", OUT)
