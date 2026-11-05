import glob
from pptx import Presentation
from pptx.util import Pt
from pptx.dml.color import RGBColor

f = [x for x in glob.glob('ppt/*.pptx') if '~$' not in x and len(Presentation(x).slides) == 20][0]
p = Presentation(f)
s = p.slides[2]
for sh in s.shapes:
    if sh.has_text_frame and '重力异常' in sh.text_frame.text and '实测' in sh.text_frame.text:
        tf = sh.text_frame
        tf.clear()
        para = tf.paragraphs[0]
        segs = [
            ('上一页说的“波长”，就是相邻波峰或波谷之间的空间距离；重力异常可看成许多不同波长成分的叠加。', False),
            ('网格间距 Δx 决定能采到的最短波长 ', False),
            ('λ_min=2Δx', True),
            ('，也就是网格分辨率：', False),
            ('1′ 网格', True),
            ('只能稳定表示约 ', False),
            ('12–20 km', True),
            (' 以上，', False),
            ('5–20 km 短波', True),
            (' 要靠 ', False),
            ('15″', True),
            (' 重建。', False),
        ]
        for txt, red in segs:
            r = para.add_run()
            r.text = txt
            r.font.size = Pt(16)
            r.font.bold = True
            r.font.name = 'Microsoft YaHei'
            r.font.color.rgb = RGBColor(0xC0, 0x39, 0x2B) if red else RGBColor(0x1A, 0x1A, 0x1A)
        para.line_spacing = 1.2
        break
p.save(f)
print('ok')
