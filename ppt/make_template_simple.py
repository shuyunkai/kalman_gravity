"""极简模板：上方深色一段文案，下方一张图。"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).parent.parent
FIG = ROOT / "output_real" / "three_parts" / "fig_paper_method_flow.png"
OUT = Path(__file__).parent / "汇报模板_一页_简洁.pptx"

NAVY = RGBColor(0x0B, 0x25, 0x45)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
s = prs.slides.add_slide(prs.slide_layouts[6])

# 上方深色区
from pptx.enum.shapes import MSO_SHAPE
rect = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0,
                          prs.slide_width, Inches(2.35))
rect.fill.solid(); rect.fill.fore_color.rgb = NAVY
rect.line.fill.background()

# 一段文案，统一字号
tb = s.shapes.add_textbox(Inches(0.85), Inches(0.55),
                          Inches(11.65), Inches(1.35))
tf = tb.text_frame; tf.word_wrap = True
tf.vertical_anchor = MSO_ANCHOR.MIDDLE
tf.margin_left = 0; tf.margin_right = 0
tf.margin_top = 0; tf.margin_bottom = 0
p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
p.line_spacing = 1.55
r = p.add_run()
r.text = ("空间相关自适应卡尔曼滤波把卫星 1′ 块观测、船测点更新和地形先验"
          "放进同一套 15″ 状态，用协方差 P∘L 引入空间相关，并用新息在线"
          "估计 R_sat，重建高分辨率重力场。")
r.font.size = Pt(19)
r.font.color.rgb = WHITE
r.font.name = "Microsoft YaHei"

# 下方图片
s.shapes.add_picture(str(FIG), Inches(0.9), Inches(2.55),
                     width=Inches(11.55))

prs.save(OUT)
print("saved", OUT)
