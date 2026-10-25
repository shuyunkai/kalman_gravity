"""单标题：一个标题 + 一段正文（含结论）+ 图。"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).parent.parent
FIG = ROOT / "output_real" / "three_parts" / "fig_paper_method_flow.png"
OUT = Path(__file__).parent / "汇报模板_单标题.pptx"

NAVY = RGBColor(0x0B, 0x25, 0x45)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PALE = RGBColor(0xC9, 0xD9, 0xEA)

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


def text(x, y, w, h, txt, size, color, bold=False, spacing=1.35):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = 0; tf.margin_right = 0
    tf.margin_top = 0; tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.TOP
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
    p.line_spacing = spacing
    r = p.add_run(); r.text = txt
    r.font.size = Pt(size); r.font.color.rgb = color
    r.font.bold = bold; r.font.name = "Microsoft YaHei"


# 深色区：一个标题 + 一段正文
rect(0, 0, 13.333, 2.95, NAVY)

text(0.7, 0.55, 11.9, 0.6, "2.3 空间相关自适应卡尔曼滤波",
     28, WHITE, True)

text(0.7, 1.45, 12.1, 1.35,
     "针对卫星 1′ 足迹与船测稀疏，把 15″ 重力场作为状态、"
     "卫星块平均与船测点作为观测；用 P∘L 修正协方差引入空间相关，"
     "用新息自适应估计 R_sat，重建高分辨率重力场。同一套 15″ 场"
     "进入断裂识别，5 km 覆盖 61.6%。",
     17, PALE, False, spacing=1.45)

# 下方只放图
s.shapes.add_picture(str(FIG), Inches(1.45), Inches(3.12),
                     width=Inches(10.4))

prs.save(OUT)
print("saved", OUT)
