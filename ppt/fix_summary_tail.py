"""按 shape id 修正 9.1 的主线卡片与底部结论。"""
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

P = r"D:\gravity\kalman_gravity\ppt\卡尔曼滤波重建地球重力场_含研究内容.pptx"
FIX = {
    13: "留出验证 3.59 / 4.84 mGal；三个验证区改善 9.4–10.7%",
    25: "KF 场补上 1′ 数据的短波短板：断裂与反演同时受益",
}
INTRO = ("同样算法下，KF 场补上了 1′ 数据在 5–50 km 的短波短板，两个应用都因此受益。")
C1 = [("断裂：1′ 数据平滑掉 5–20 km 构造；KF 场补回后，5 km 覆盖 ", False),
      ("61.6% vs 10.1%", True), ("，中位距 ", False),
      ("3.7 vs 11.7 km", True), ("。", False)]
C2 = [("反演：1′ 数据在 10–50 km 精度不足；KF 场把 RMSE 从 ", False),
      ("131 m 降到 124 m", True), ("、corr ", False),
      ("0.77→0.82", True), ("；20–200 km 两者相当。", False)]

prs = Presentation(P)
s = None
for sl in prs.slides:
    for sh in sl.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip().startswith("9.1"):
            s = sl
            break
    if s is not None:
        break
if s is None:
    raise SystemExit("9.1 not found")

n = 0
for sh in s.shapes:
    if sh.shape_id in FIX and sh.has_text_frame:
        para = sh.text_frame.paragraphs[0]
        for r in para.runs:
            r.text = ""
        if para.runs:
            para.runs[0].text = FIX[sh.shape_id]
        n += 1

# 卡片上方的文字总结：先删旧版，再新增一个
MARK = ("重建引擎", "本项目用卡尔曼滤波", "同样算法下")
for sh in list(s.shapes):
    if sh.has_text_frame and any(k in sh.text_frame.text for k in MARK):
        sh._element.getparent().remove(sh._element)
tb = s.shapes.add_textbox(Inches(0.75), Inches(1.52),
                          Inches(11.9), Inches(0.54))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
r = p.add_run()
r.text = INTRO
r.font.size = Pt(11)
r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

# 卡片下方的文字总结
for sh in s.shapes:
    if sh.shape_id == 22 and sh.has_text_frame:
        tf = sh.text_frame
        tf.clear()
        sh.top = Inches(5.32)
        sh.height = Inches(1.05)

        def add_rich(para, segs, color):
            for txt, bold in segs:
                r = para.add_run()
                r.text = txt
                r.font.size = Pt(10.5)
                r.font.bold = bold
                r.font.color.rgb = color

        p1 = tf.paragraphs[0]
        p1.space_after = Pt(6)
        add_rich(p1, C1, RGBColor(0x33, 0x33, 0x33))
        p2 = tf.add_paragraph()
        p2.space_after = Pt(0)
        add_rich(p2, C2, RGBColor(0x33, 0x33, 0x33))
        n += 1

prs.save(P)
print("fixed", n)
