"""用模板第 1 页做封面，替换标题与汇报人。"""
import subprocess
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN

ROOT = Path(__file__).parent.parent
TEMPLATE = ROOT.parent / "模板3_石大深蓝豪华风格(1).pptx"
OUT = ROOT / "ppt" / "封面_石大.pptx"
TMP = ROOT / "ppt" / "_cover_src.pptx"

# 用 COM 只保留模板封面一页
code = f'''
import win32com.client as w
a=w.Dispatch("PowerPoint.Application")
p=a.Presentations.Open(r"{TEMPLATE}", WithWindow=False)
for i in range(p.Slides.Count,0,-1):
    if i!=1: p.Slides(i).Delete()
p.SaveAs(r"{TMP}")
p.Close(); a.Quit()
'''
subprocess.run([sys.executable, "-c", code], check=True)

prs = Presentation(TMP)
s = prs.slides[0]
GOLD = RGBColor(0xD9, 0xB4, 0x5B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def replace(shape, lines, color, size=None):
    tf = shape.text_frame
    old = tf.paragraphs[0].runs[0] if tf.paragraphs[0].runs else None
    fs = size or (old.font.size.pt if old and old.font.size else 24)
    tf.clear()
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = tf.paragraphs[0].alignment
        r = p.add_run(); r.text = line
        r.font.size = Pt(fs); r.font.bold = True
        r.font.name = "Microsoft YaHei"; r.font.color.rgb = color


for sh in s.shapes:
    if not sh.has_text_frame:
        continue
    t = sh.text.strip()
    top = sh.top / 914400
    if "2021" in t:
        replace(sh, ["15″ 高分辨率重力场"], GOLD)
    elif "数字岩石物理" in t:
        replace(sh, ["基于相干保持卡尔曼滤波的", "地球重力场重建"], GOLD, size=34)
    elif "汇报人" in t:
        replace(sh, ["汇报人：____________", "", "时    间：2026 年 10 月"], WHITE, size=22)

# 副标题
tb = s.shapes.add_textbox(Inches(3.0), Inches(4.05), Inches(8.2), Inches(0.5))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
r = p.add_run(); r.text = "——船测—卫星重力融合与断裂应用"
r.font.size = Pt(20); r.font.name = "Microsoft YaHei"
r.font.color.rgb = GOLD

prs.save(OUT)
TMP.unlink(missing_ok=True)
print("saved", OUT)
