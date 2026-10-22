"""按 shape id 修正 9.1 的主线卡片与底部结论。"""
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

P = r"D:\gravity\kalman_gravity\ppt\卡尔曼滤波重建地球重力场_含研究内容.pptx"
FIX = {
    13: "留出验证 3.59 / 4.84 mGal；三个验证区改善 9.4–10.7%",
    25: "KF 场 → 断裂 + 反演：5–50 km 全面优于常规数据",
}
INTRO = ("本项目把卡尔曼滤波作为重建引擎：卫星测高、船载重力与地形先验进入同一套"
         "状态—观测方程，递推得到 15″ 场，再用于断裂识别与海底地形反演。")
C1 = ("同一套带通/THG 与 Parker 下，KF 场在 5–50 km 全面优于 1′ 卫星重力直接插值："
      "断裂 5 km 覆盖 61.6% vs 10.1%，反演 10–50 km RMSE 124 vs 131 m。")
C2 = ("增益来自重建后的场，而不是下游算法；5–20 km 部分来自地形先验，"
      "1–5 km 仍受公开重力与断层编目限制。")

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

# 卡片上方的文字总结
if not any(sh.has_text_frame and "重建引擎" in sh.text_frame.text for sh in s.shapes):
    tb = s.shapes.add_textbox(Inches(0.75), Inches(1.52),
                              Inches(11.9), Inches(0.54))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = INTRO
    r.font.size = Pt(11)
    r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
for sh in s.shapes:
    if sh.has_text_frame and "重建引擎" in sh.text_frame.text:
        sh.top = Inches(1.52)
        sh.height = Inches(0.54)

# 卡片下方的文字总结
for sh in s.shapes:
    if sh.shape_id == 22 and sh.has_text_frame:
        tf = sh.text_frame
        tf.clear()
        sh.height = Inches(1.28)
        p1 = tf.paragraphs[0]
        r1 = p1.add_run()
        r1.text = C1
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = C2
        r2.font.size = Pt(9.5)
        r2.font.color.rgb = RGBColor(0x5B, 0x65, 0x73)
        n += 1

prs.save(P)
print("fixed", n)
