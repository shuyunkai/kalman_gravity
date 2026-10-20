from pptx import Presentation

P = r"D:\gravity\kalman_gravity\ppt\卡尔曼滤波重建地球重力场_含研究内容.pptx"
prs = Presentation(P)
s = prs.slides[15]

rules = [
    ("必须写明口径",
     "同一份 15″ 结果在两种口径下相差 0.02–1.03 mGal，排名与改善幅度一致。"),
    ("报告中必须注明",
     "两种口径差 0.02–1.03 mGal，结论对口径不敏感"),
    ("国际上模型间对比",
     "1′ 块口径：模型块均值对船载块均值，用于模型间对比"),
]
for sh in s.shapes:
    if not sh.has_text_frame:
        continue
    for p in sh.text_frame.paragraphs:
        full = "".join(r.text for r in p.runs)
        for k, v in rules:
            if k in full:
                for r in p.runs:
                    r.text = ""
                if p.runs:
                    p.runs[0].text = v
                break
prs.save(P)
print("done")
