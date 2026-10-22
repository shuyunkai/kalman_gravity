"""按最终叙事顺序重编号：6.2/8.8/7.2/8.5/8.7/8.9/7.3/8.6/9.1 → 9.x/10。"""
import re

from pptx import Presentation

P = r"D:\gravity\kalman_gravity\ppt\卡尔曼滤波重建地球重力场_含研究内容.pptx"

MAP = {
    "6.2": "9.1",
    "8.8": "9.2",
    "7.2": "9.3",
    "8.5": "9.4",
    "8.7": "9.5",
    "8.9": "9.6",
    "7.3": "9.7",
    "8.6": "9.8",
    "9.1": "10",
}

prs = Presentation(P)
titles = []
for sl in prs.slides:
    for sh in sl.shapes:
        if not sh.has_text_frame:
            continue
        t = sh.text_frame.text.strip()
        m = re.match(r"^(\d+(?:\.\d+)?)(\s*[｜|])", t)
        if m and m.group(1) in MAP:
            titles.append((sh, m.group(1), t[len(m.group(1)):]))
            break


def set_title(sh, text):
    para = sh.text_frame.paragraphs[0]
    for r in para.runs:
        r.text = ""
    if para.runs:
        para.runs[0].text = text
    else:
        para.add_run().text = text


codes = {code for _, code, _ in titles}
if not codes & {"6.2", "8.8", "7.2", "8.5", "8.7", "8.9", "7.3", "8.6"}:
    print("already renumbered")
    raise SystemExit

for i, (sh, code, rest) in enumerate(titles):
    set_title(sh, f"§TMP{i}§{rest}")

for (sh, code, rest), i in zip(titles, range(len(titles))):
    set_title(sh, f"{MAP[code]}{rest}")

prs.save(P)
for i, (sh, code, rest) in enumerate(titles, 1):
    print(i, MAP[code], rest[:24])
