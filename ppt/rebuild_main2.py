"""用 14 页原版 + 1.2 页重建 15 页主 deck。"""
import shutil
import win32com.client as win32

PPT = r"D:\gravity\kalman_gravity\ppt"
SRC = PPT + r"\重力场重建_杰青风格汇报.pptx"
OUT = PPT + r"\卡尔曼滤波重建地球重力场_含研究内容.pptx"
MAIN = PPT + r"\卡尔曼滤波重建地球重力场.pptx"

shutil.copyfile(SRC, MAIN)
app = win32.Dispatch("PowerPoint.Application")
prs = app.Presentations.Open(MAIN, WithWindow=False)
prs.Slides.InsertFromFile(OUT, 2, 3, 3)  # OUT 第 3 页 = 1.2
print("slides", prs.Slides.Count)
for i in range(1, prs.Slides.Count + 1):
    t = ""
    for sh in prs.Slides(i).Shapes:
        try:
            if sh.HasTextFrame and sh.TextFrame.HasText:
                t = sh.TextFrame.TextRange.Text.split("\r")[0]
                break
        except Exception:
            pass
    print(i, t[:24])
prs.Save()
prs.Close()
app.Quit()
