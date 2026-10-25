"""常规单页：标题 + 一段文案 + 图片 + 页脚页码。"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).parent.parent
FIG = ROOT / "output_real" / "three_parts" / "fig_paper_method_flow.png"
OUT = Path(__file__).parent / "汇报模板_一页_常规.pptx"

NAVY = RGBColor(0x0B, 0x25, 0x45)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PALE = RGBColor(0xC9, 0xD9, 0xEA)
GREY = RGBColor(0x5B, 0x65, 0x73)
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


def text(x, y, w, h, txt, size, color, bold=False, spacing=1.2,
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


# 深色标题区
rect(0, 0, 13.333, 2.68, NAVY)
rect(0, 2.68, 13.333, 0.05, RED)

text(0.65, 0.26, 8.0, 0.3, "二、研究问题与学术思路", 12.5, PALE)
text(11.2, 0.26, 1.5, 0.3, "2026.10", 11, PALE, align=PP_ALIGN.RIGHT)

text(0.65, 0.62, 12.0, 0.95,
     "基于空间相关自适应卡尔曼滤波的 15″ 地球重力场重建",
     25, WHITE, True, spacing=1.1)

text(0.65, 1.68, 11.9, 0.85,
     "把卫星 1′ 块观测、船测点更新和地形先验放进同一套 15″ 状态，"
     "用协方差 P∘L 引入空间相关，并用新息在线估计 R_sat，重建高分辨率重力场。",
     13, WHITE, False, spacing=1.4)

# 图片
s.shapes.add_picture(str(FIG), Inches(1.05), Inches(2.92),
                     width=Inches(11.2))

# 页脚
rect(0.65, 7.18, 12.0, 0.012, RGBColor(0xC9, 0xD9, 0xEA))
text(0.65, 7.24, 9.0, 0.22,
     "汇报人：舒云恺　｜　空间相关自适应卡尔曼滤波", 8.5, GREY)
text(11.9, 7.24, 0.8, 0.22, "3 / 24", 8.5, GREY, align=PP_ALIGN.RIGHT)

prs.save(OUT)
print("saved", OUT)
