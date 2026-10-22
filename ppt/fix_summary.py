"""按“KF 重建主线 + 两项应用”改写 9.1 总结页。"""
from pptx import Presentation

P = r"D:\gravity\kalman_gravity\ppt\卡尔曼滤波重建地球重力场_含研究内容.pptx"

RULES = [
    ("三类观测递推", "KF 重建 15″ 场是主线；断层与反演是这份场的两项应用。"),
    ("以 15″ 重力场为状态", "主线：重建 15″ 场"),
    ("三类观测写进同一套方程",
     "三类观测同解；留出验证 MR22-04 3.59 mGal、JODC 4.84 mGal"),
    ("把观测写成同一套方程", "应用一：断裂"),
    ("五种现有做法",
     "5–20 km 带通 + THG；GM9 5 km 覆盖 61.6% vs SSV 10.1%，中位距 3.7 vs 11.7 km"),
    ("三个区改善", "应用二：反演"),
    ("12.7 万个验证点",
     "Parker 20–200 km；20–50 km 112 m，10–50 km 124 vs 131 m，corr 0.82 vs 0.77"),
    ("下一步", "边界：增益在 5–50 km；1–5 km 受公开数据限制；5–20 km 含地形先验，需注明"),
    ("俯冲带剩余重力格网", "建议：用公开海域断层编目与船测加密，分离地形先验的贡献。"),
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
