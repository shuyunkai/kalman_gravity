"""清空所有演讲者备注：备注只在 PowerPoint 编辑视图显示，换软件打开会丢，统一去掉。"""
from pptx import Presentation

P = r"D:\gravity\kalman_gravity\ppt\卡尔曼滤波重建地球重力场_含研究内容.pptx"

prs = Presentation(P)
n = 0
for s in prs.slides:
    if s.has_notes_slide:
        tf = s.notes_slide.notes_text_frame
        if tf.text.strip():
            tf.clear()
            n += 1
prs.save(P)
print("cleared", n)
