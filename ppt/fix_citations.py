"""去掉 2.1 页“代表工作”后面的论文出处，只保留模型名。"""
from pptx import Presentation

P = r"D:\gravity\kalman_gravity\ppt\卡尔曼滤波重建地球重力场_含研究内容.pptx"

prs = Presentation(P)
s = prs.slides[4]
n = 0
for sh in s.shapes:
    if not sh.has_text_frame:
        continue
    for para in sh.text_frame.paragraphs:
        runs = para.runs
        if len(runs) >= 2 and runs[0].text.endswith("　"):
            for r in runs[1:]:
                r._r.getparent().remove(r._r)
            n += 1
prs.save(P)
print("cleaned", n)
