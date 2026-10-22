"""按 shape id 修正 9.1 的主线卡片与底部结论。"""
from pptx import Presentation

P = r"D:\gravity\kalman_gravity\ppt\卡尔曼滤波重建地球重力场_含研究内容.pptx"
FIX = {
    13: "留出验证 3.59 / 4.84 mGal；三个验证区改善 9.4–10.7%",
    25: "KF 场 → 断裂 + 反演：5–50 km 全面优于常规数据",
}

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
prs.save(P)
print("fixed", n)
