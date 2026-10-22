"""2.1 页：去掉论文出处，给 1′ 模型补产品属性。"""
from pptx import Presentation
from pptx.oxml.ns import qn

P = r"D:\gravity\kalman_gravity\ppt\卡尔曼滤波重建地球重力场_含研究内容.pptx"

ATTR = {
    "SIO 系列": "SIO 系列　全球 1′ 海洋重力",
    "SDUST2022GRA": "SDUST2022GRA　中国近海 1′",
    "NSOAS24": "NSOAS24　全球 1′ 重力",
    "DTU21GRA": "DTU21GRA　全球 1′ 海洋重力",
}

prs = Presentation(P)
s = None
for sl in prs.slides:
    for sh in sl.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip().startswith("2.1"):
            s = sl
            break
    if s is not None:
        break
if s is None:
    raise SystemExit("slide 2.1 not found")

n = 0
for sh in s.shapes:
    if not sh.has_text_frame:
        continue
    for para in sh.text_frame.paragraphs:
        runs = para.runs
        if runs and runs[0].text.endswith("　"):
            name = runs[0].text.strip()
            if name in ATTR:
                runs[0].text = ATTR[name]
            for r in runs[1:]:
                r._r.getparent().remove(r._r)
            n += 1
prs.save(P)
print("updated", n)

# 2.2：删掉三条方法下面的论文出处（整段独立文本框）
import re
s22 = None
for sl in prs.slides:
    for sh in sl.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip().startswith("2.2"):
            s22 = sl
            break
    if s22 is not None:
        break
if s22 is not None:
    cit = re.compile(r"等[,，].*(19|20)\d{2}")
    DESC = [
        ("Zhu", "用成对样本学习重力→地形映射"),
        ("Chen", "按波长分带融合测高与船测"),
        ("刘乾坤", "联合解算测高与船测重力"),
    ]
    m = 0
    for t in list(s22._element.iter(qn("a:t"))):
        txt = t.text or ""
        if len(txt) < 80 and cit.search(txt):
            for key, desc in DESC:
                if key in txt:
                    t.text = desc
                    break
            m += 1
    print("descriptions replaced", m)
    prs.save(P)
from pptx.oxml.ns import qn
