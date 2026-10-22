"""按“KF 重建主线 + 两个应用”重排：精度结果之后先流程，再断层、反演。"""
from pptx import Presentation

P = r"D:\gravity\kalman_gravity\ppt\卡尔曼滤波重建地球重力场_含研究内容.pptx"

ORDER = ["1.1", "1.2", "1.3", "2.1", "2.2", "3.1", "4.1", "5.1",
         "6.1", "7.1", "8.1", "8.2", "8.3", "8.4", "6.2",
         "8.8", "7.2", "8.5", "8.7", "8.9", "7.3", "8.6", "9.1"]

prs = Presentation(P)
slides = list(prs.slides)
sld_ids = list(prs.slides._sldIdLst)
id_map = {id(s): sid for s, sid in zip(slides, sld_ids)}


def title(s):
    for sh in s.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip():
            return sh.text_frame.text.strip().split("\n")[0]
    return ""


cover = [s for s in slides if title(s) and not title(s)[0].isdigit()][0]
ordered = [cover]
used = {id(cover)}
for pref in ORDER:
    for s in slides:
        if id(s) in used:
            continue
        if title(s).startswith(pref):
            ordered.append(s)
            used.add(id(s))
            break
    else:
        print("missing", pref)
for s in slides:
    if id(s) not in used:
        ordered.append(s)
        used.add(id(s))

for s in ordered:
    prs.slides._sldIdLst.append(id_map[id(s)])
prs.save(P)
for i, s in enumerate(ordered, 1):
    print(i, title(s)[:24])
