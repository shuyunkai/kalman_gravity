"""用石大单页母版做封面。"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).parent.parent
SRC = ROOT / "ppt" / "模板_石大_单页母版.pptx"
OUT = ROOT / "ppt" / "封面_石大.pptx"

GOLD = RGBColor(0xC9, 0xA2, 0x27)
NAVY = RGBColor(0x0B, 0x1F, 0x3A)
GREY = RGBColor(0x5B, 0x65, 0x73)

prs = Presentation(SRC)
s = prs.slides[0]
for sh in list(s.shapes):
    top = sh.top / 914400 if sh.top is not None else 0
    if 1.0 < top < 7.0:
        sh._element.getparent().remove(sh._element)
    elif sh.shape_type == 17 and top < 1.0 and sh.width and sh.width / 914400 > 3:
        sh._element.getparent().remove(sh._element)


def text(y, h, txt, size, color, bold=True, align=PP_ALIGN.CENTER):
    tb = s.shapes.add_textbox(Inches(0.8), Inches(y), Inches(11.7), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = 0; tf.margin_right = 0
    tf.margin_top = 0; tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = txt
    r.font.size = Pt(size); r.font.bold = bold
    r.font.color.rgb = color; r.font.name = "Microsoft YaHei"


text(1.75, 1.0, "基于相干保持卡尔曼滤波的", 30, NAVY)
text(2.55, 1.0, "15″ 地球重力场重建", 40, GOLD)
text(3.75, 0.6, "——船测与卫星重力融合及断裂应用", 18, GREY, bold=False)
text(5.35, 0.5, "汇报人：____________", 14, GREY, bold=False)
text(5.85, 0.5, "2026 年 10 月", 14, GREY, bold=False)
prs.save(OUT)
print("saved", OUT)
