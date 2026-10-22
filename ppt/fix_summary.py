"""按“KF 重建主线 + 两项应用”改写 9.1 总结页。"""
from pptx import Presentation

P = r"D:\gravity\kalman_gravity\ppt\卡尔曼滤波重建地球重力场_含研究内容.pptx"

RULES = [
    ("三类观测递推", "KF 把三类观测融成 15″ 场；这份场让断裂与反演同时超过常规数据。"),
    ("以 15″ 重力场为状态", "主线：15″ 场"),
    ("三类观测写进同一套方程",
     "留出验证 3.59 / 4.84 mGal；三个验证区改善 9.4–10.7%"),
    ("把观测写成同一套方程", "应用一：断裂"),
    ("五种现有做法",
     "5 km 覆盖 61.6% vs 10.1%（6 倍）；中位距 3.7 vs 11.7 km"),
    ("三个区改善", "应用二：反演"),
    ("12.7 万个验证点",
     "10–50 km RMSE 124 vs 131 m，corr 0.82 vs 0.77；20–200 km 372 m"),
    ("下一步", "5–50 km 全面占优；1–5 km 受公开数据限制"),
    ("俯冲带剩余重力格网", "说明：5–20 km 的增益部分来自地形先验。"),
]

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
    if not sh.has_text_frame:
        continue
    for para in sh.text_frame.paragraphs:
        full = "".join(r.text for r in para.runs)
        for key, val in RULES:
            if key in full:
                for r in para.runs:
                    r.text = ""
                if para.runs:
                    para.runs[0].text = val
                n += 1
                break
prs.save(P)
print("updated", n)
