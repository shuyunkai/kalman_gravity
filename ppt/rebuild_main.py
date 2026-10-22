"""从当前 OUT 恢复 15 页主 deck：封面…8.4 + 总结页。"""
from pptx import Presentation

PPT = r"D:\gravity\kalman_gravity\ppt"
OUT = PPT + r"\卡尔曼滤波重建地球重力场_含研究内容.pptx"
MAIN = PPT + r"\卡尔曼滤波重建地球重力场.pptx"

prs = Presentation(OUT)
ids = list(prs.slides._sldIdLst)
print("current", len(ids))
keep = set(range(14)) | {23}
for i, sid in enumerate(ids):
    if i not in keep:
        prs.slides._sldIdLst.remove(sid)

# 总结页标题恢复为 9.1
s = prs.slides[14]
for sh in s.shapes:
    if sh.has_text_frame:
        t = sh.text_frame.text.strip()
        if t.startswith("10"):
            para = sh.text_frame.paragraphs[0]
            for r in para.runs:
                r.text = ""
            if para.runs:
                para.runs[0].text = "9.1 ｜ 总结"
            else:
                para.add_run().text = "9.1 ｜ 总结"
            break

prs.save(MAIN)
print("main slides", len(prs.slides))
