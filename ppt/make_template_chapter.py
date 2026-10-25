"""章标题页：顶部章标题，下面本节标题+说明+图+结论条。"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).parent.parent
FIG = ROOT / "output_real" / "three_parts" / "fig_paper_method_flow.png"
OUT = Path(__file__).parent / "汇报模板_章标题页.pptx"

NAVY = RGBColor(0x0B, 0x25, 0x45)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PALE = RGBColor(0xC9, 0xD9, 0xEA)
GREY = RGBColor(0x33, 0x33, 0x33)
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


# 章标题
rect(0, 0, 13.333, 1.78, NAVY)
text(0.7, 0.38, 11.9, 1.0, "二、研究问题与学术思路",
     30, WHITE, True, spacing=1.05)

# 本节标题
text(0.7, 1.98, 11.9, 0.42, "2.3 空间相关自适应卡尔曼滤波",
     19, NAVY, True)

# 本节说明
text(0.7, 2.46, 11.9, 0.72,
     "针对卫星 1′ 足迹与船测稀疏，把 15″ 重力场作为状态、"
     "卫星块平均与船测点作为观测；用 P∘L 修正协方差引入空间相关，"
     "用新息自适应估计 R_sat。",
     12, GREY, False, spacing=1.4)

# 图
s.shapes.add_picture(str(FIG), Inches(1.75), Inches(3.28),
                     width=Inches(9.8))

# 结论条
rect(0.7, 6.88, 11.9, 0.35, RED)
text(0.85, 6.92, 11.6, 0.28,
     "结论：同一套 15″ 场进入断裂识别，5 km 覆盖 61.6%",
     11, WHITE, True)

# 页脚
text(0.7, 7.28, 8.0, 0.2,
     "汇报人：舒云恺　｜　空间相关自适应卡尔曼滤波", 8, RGBColor(0x5B, 0x65, 0x73))
text(12.0, 7.28, 0.7, 0.2, "3 / 24", 8, RGBColor(0x5B, 0x65, 0x73),
     align=PP_ALIGN.RIGHT)

prs.save(OUT)
print("saved", OUT)
