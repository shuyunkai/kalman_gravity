"""按 shape id 修正 9.1 的主线卡片与底部结论。"""
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

P = r"D:\gravity\kalman_gravity\ppt\卡尔曼滤波重建地球重力场_含研究内容.pptx"
FIX = {
    13: "留出验证 3.59 / 4.84 mGal；三个验证区改善 9.4–10.7%",
    25: "KF 场 → 断裂 + 反演：5–50 km 全面优于常规数据",
}
INTRO = ("本项目用卡尔曼滤波把卫星测高、船载重力与地形先验融成 15″ 场；断层与反演都直接"
         "使用这份场，下游算法与对比实验完全一致，因此差异只来自重力场本身。")
C1 = ("同样用 5–20 km 带通 + THG，KF 场的线性体在 5 km 内覆盖 61.6% 的留出断层，"
      "SSV 插值只有 10.1%；中位距 3.7 km vs 11.7 km——KF 场把被 1′ 网格平滑掉的短波"
      "构造重新显出来，断裂定位更锐、更贴断层。")
C2 = ("同样用 Parker，10–50 km RMSE 124 m vs 131 m、corr 0.82 vs 0.77；但 20–200 km "
      "两者持平（372 vs 376 m），说明 KF 的增益集中在 5–50 km，长波仍由卫星重力主导。")
C3 = ("工程含义：KF 场的价值在短波构造应用（断裂、5–50 km 地形），而不是全频段替代"
      "卫星重力；5–20 km 的增益部分来自地形先验，1–5 km 受公开数据限制。")

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
        tf = sh.text_frame
        p = tf.paragraphs[0]
        for r in p.runs:
            r.text = ""
        r = p.runs[0] if p.runs else p.add_run()
        r.text = INTRO
        r.font.size = Pt(11)
        r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

# 卡片下方的文字总结
for sh in s.shapes:
    if sh.shape_id == 22 and sh.has_text_frame:
        tf = sh.text_frame
        tf.clear()
        sh.top = Inches(5.12)
        sh.height = Inches(1.62)
        p1 = tf.paragraphs[0]
        p1.space_after = Pt(5)
        r1 = p1.add_run()
        r1.text = C1
        r1.font.size = Pt(10)
        r1.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        for txt, size in ((C2, 10), (C3, 9)):
            p = tf.add_paragraph()
            p.space_after = Pt(5)
            r = p.add_run()
            r.text = txt
            r.font.size = Pt(size)
            r.font.color.rgb = RGBColor(0x5B, 0x65, 0x73)
        n += 1

prs.save(P)
print("fixed", n)
