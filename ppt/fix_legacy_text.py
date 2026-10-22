from pptx import Presentation

P = r"D:\gravity\kalman_gravity\ppt\卡尔曼滤波重建地球重力场_含研究内容.pptx"
prs = Presentation(P)

s = None
for sl in prs.slides:
    for sh in sl.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip().startswith("8.3"):
            s = sl
            break
    if s is not None:
        break
if s is None:
    raise SystemExit("slide 8.3 not found")

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

# 8.4 说明句缩短，避免句号换行
for sl in prs.slides:
    for sh in sl.shapes:
        if sh.has_text_frame and "相对卫星模型的改善" in sh.text_frame.text:
            p = sh.text_frame.paragraphs[0]
            for r in p.runs:
                r.text = ""
            txt = ("相对卫星模型的改善（0.15–0.50 mGal）大于相对频域融合"
                   "（0.11–0.40 mGal）——频域融合已压缩短波，本方法仍有增益。")
            if p.runs:
                p.runs[0].text = txt
            else:
                p.add_run().text = txt

# 清掉单独成段的句号
for sl in prs.slides:
    for sh in list(sl.shapes):
        if not sh.has_text_frame:
            continue
        if sh.text_frame.text.strip() in ("。", "."):
            sh._element.getparent().remove(sh._element)
            continue
        ps = sh.text_frame.paragraphs
        for p in list(ps):
            txt = "".join(r.text for r in p.runs).strip()
            if txt in ("。", ".") and len(ps) > 1:
                p._p.getparent().remove(p._p)

prs.save(P)
print("done")

