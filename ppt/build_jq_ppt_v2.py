"""按新流程重建汇报 PPT。

流程（用户指定）:
  1.1 什么是地球重力场 → 2.1/2.2 主流方法(1′ 与 15″ 两部分) → 3.1 存在的问题
  → 4.1 卡尔曼滤波能解决什么 → 5.1 什么是卡尔曼滤波 → 6.1 本项目的思路
  → 7.1 本项目的流程 → 8.1 结果验证 → 9.1 总结

版式照参考模板: 整宽深色页眉带「N.N ｜ 标题」+ 章节药丸 + 说明行(关键词标红加粗)
+ 内容面板/图 + 底部整宽结论带(关键词标红)。
"""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(r"D:\gravity\ppt_jq")
FIGS = ROOT / "figs"
OUT = ROOT / "重力场重建_杰青风格汇报.pptx"

NAVY = RGBColor(0x16, 0x36, 0x5C)
NAVY_D = RGBColor(0x0E, 0x27, 0x42)
MID = RGBColor(0x4A, 0x7A, 0xA7)
LIGHT = RGBColor(0xD8, 0xE0, 0xEC)
GREY = RGBColor(0x5B, 0x65, 0x73)
DARK = RGBColor(0x1C, 0x24, 0x30)
PALE = RGBColor(0xEE, 0xF3, 0xFA)
RED = RGBColor(0xC0, 0x39, 0x2B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
HELPER = RGBColor(0x8A, 0x93, 0xA0)
RED_TINT = RGBColor(0xFB, 0xEE, 0xEC)
FONT = "微软雅黑"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def style(run, size, color, bold):
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.name = FONT


def rect(s, x, y, w, h, color):
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = color
    sh.line.fill.background(); sh.shadow.inherit = False
    return sh


def textbox(s, x, y, w, h, paras, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=None):
    """paras: 每段可以是 (文本,字号,颜色,粗) 或 [(文本,字号,颜色,粗), ...] 混排。"""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    first = True
    for para in paras:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = align
        if spacing:
            p.line_spacing = spacing
        runs = para if isinstance(para, list) else [para]
        for txt, size, color, bold in runs:
            r = p.add_run(); style(r, size, color, bold); r.text = txt
    return tb


def strip(s, segs):
    """底部整宽结论带；segs 为混排片段。"""
    if isinstance(segs, tuple):
        segs = [segs]
    rect(s, 0.42, 6.74, 12.49, 0.56, PALE)
    rect(s, 0.42, 6.74, 0.075, 0.56, RED)
    textbox(s, 0.78, 6.85, 11.95, 0.40,
            [[(t, 15, RED if c == "r" else NAVY, True) for t, c in segs]])


def banner(s, code, main, section, sub):
    rect(s, 0, 0, 13.333, 0.68, NAVY)
    rect(s, 0, 0.68, 13.333, 0.042, RED)
    textbox(s, 0.42, 0.14, 9.8, 0.44, [(f"{code} ｜ {main}", 20, WHITE, True)])
    textbox(s, 10.1, 0.19, 2.8, 0.32,
            [("卡尔曼滤波重建地球重力场", 9.5, LIGHT, False)], align=PP_ALIGN.RIGHT)
    rect(s, 0.42, 0.84, 0.92, 0.30, NAVY)
    textbox(s, 0.42, 0.885, 0.92, 0.24, [(section, 10.5, WHITE, True)],
            align=PP_ALIGN.CENTER)
    textbox(s, 1.50, 0.815, 11.3, 0.62, sub, spacing=1.25)
    rect(s, 0.42, 1.48, 12.49, 0.010, LIGHT)


def new_slide(code, main, section, sub):
    s = prs.slides.add_slide(BLANK)
    banner(s, code, main, section, sub)
    return s


def picture(s, fname, top, max_w, max_h, left=None):
    p = FIGS / fname
    iw, ih = Image.open(p).size
    w, h = max_w, max_w * ih / iw
    if h > max_h:
        h, w = max_h, max_h * iw / ih
    x = (13.333 - w) / 2 if left is None else left
    s.shapes.add_picture(str(p), Inches(x), Inches(top), Inches(w), Inches(h))


def bullets(s, x, y, w, items, size=12.5, gap=0.34):
    """➤ 列表（参考模板的写法）。"""
    for i, (lead, rest) in enumerate(items):
        yy = y + i * gap
        textbox(s, x, yy, 0.28, 0.30, [("➤", size, RED, True)])
        textbox(s, x + 0.30, yy, w - 0.30, 0.30,
                [[(lead, size, NAVY, True), (rest, size, GREY, False)]])


def notes(s, t):
    s.notes_slide.notes_text_frame.text = t


def R(t):        # 红色强调（结论带）
    return (t, "r")


def N(t):        # 常规（结论带）
    return (t, "n")


def para(*runs):
    return list(runs)


# ============================================================ 1 封面
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, 13.333, 7.5, NAVY_D)
rect(s, 0, 0, 13.333, 0.14, NAVY)
rect(s, 0, 0.14, 13.333, 0.045, RED)
textbox(s, 0.90, 1.72, 6.70, 1.05, [("卡尔曼滤波重建地球重力场", 38, WHITE, True)])
textbox(s, 0.90, 2.92, 6.80, 0.86,
        [("从 1′ 卫星测高重力到 15″ 场：", 17, LIGHT, False),
         ("方法、实测验证与横向对比", 17, LIGHT, False)], spacing=1.25)
rect(s, 0.93, 3.96, 2.80, 0.045, RED)
for k, (big, lab) in enumerate([("15″", "重建分辨率，由 1′ 提升 4 倍"),
                               ("3", "个互不重叠的验证区"),
                               ("12.7 万", "个未参与解算的验证点")]):
    x = 0.90 + k * 2.30
    if k:
        rect(s, x - 0.18, 4.36, 0.008, 0.94, MID)
    textbox(s, x, 4.28, 2.20, 0.62, [(big, 30, WHITE, True)])
    textbox(s, x, 4.94, 2.24, 0.46, [(lab, 11, LIGHT, False)])
picture(s, "fig_cover_map.png", 1.78, 4.96, 4.60, left=7.92)
textbox(s, 7.85, 6.34, 5.00, 0.32,
        [("研究区与三个独立验证区（红框）", 10.5, HELPER, False)], align=PP_ALIGN.CENTER)
textbox(s, 0.90, 6.36, 6.60, 0.42,
        [("汇报人　＿＿＿＿＿＿　　　报告日期　＿＿＿＿＿＿", 13, HELPER, False)])
notes(s, "全场 11 页，九个章节：重力场是什么 → 两条主流路线 → 存在的问题 → "
         "卡尔曼滤波能解决什么 → 原理 → 本项目思路 → 本项目流程 → 结果验证 → 总结。")

# ============================================================ 2 什么是地球重力场
s = new_slide("1.1", "什么是地球重力场", "背景", [
    para(("按经纬度与高程可以算出任一位置的", 13.5, GREY, False),
         ("理论重力值", 13.5, RED, True),
         ("；当地", 13.5, GREY, False),
         ("实测值", 13.5, RED, True),
         ("与理论值之差，就是", 13.5, GREY, False),
         ("重力异常", 13.5, RED, True),
         (" Δg。", 13.5, GREY, False)),
    para(("它反映的是地下物质分布的不均匀 —— 海沟、洋脊、海山都会在 Δg 上留下印记。",
          12.5, GREY, False)),
])
picture(s, "fig_gravity.png", 1.70, 12.20, 4.60)
strip(s, [N("重力异常不是「重力」本身，而是"), R("它与理论值之差"), N(" —— 这正是要重建的对象")])
notes(s, "这一页只建立概念：理论重力由经纬度与高程决定（正常重力场）；"
         "实测值减去理论值得到重力异常。右侧是研究区的实际重力异常场，"
         "海沟为强负异常、洋脊为相对高值 —— 这就是后面要加密的那张场。")

# ============================================================ 3 主流方法（一）1′
s = new_slide("2.1", "主流方法（一）：1′ 卫星测高重力模型", "方法", [
    para(("最主流的路线是用", 13.5, GREY, False),
         ("卫星测高", 13.5, RED, True),
         ("得到的海面高反演重力场；目前公开的全球模型", 13.5, GREY, False),
         ("全部为 1′ 分辨率", 13.5, RED, True),
         ("。", 13.5, GREY, False)),
])
picture(s, "fig_altimetry.png", 1.58, 12.10, 2.18)
# 左：代表工作
rect(s, 0.42, 3.92, 6.00, 2.55, PALE)
rect(s, 0.42, 3.92, 0.075, 2.55, NAVY)
textbox(s, 0.75, 4.02, 5.5, 0.32, [("代表工作", 14, NAVY, True)])
bullets(s, 0.75, 4.42, 5.60, [
    ("SIO 系列　", "Sandwell 等, Science 346(6205): 65–67, 2014"),
    ("SDUST2022GRA　", "Li 等, ESSD 16: 4119–4135, 2024"),
    ("NSOAS24　", "Zhang 等, GMD 18: 1221–1239, 2025"),
    ("DTU21GRA　", "DTU Space 数据产品（1′）"),
], size=11, gap=0.42)
# 右：这些产品的共同特征（1′ 模型已是成熟产品，不在此页讨论其问题）
rect(s, 6.60, 3.92, 6.31, 2.55, PALE)
rect(s, 6.60, 3.92, 0.075, 2.55, MID)
textbox(s, 6.93, 4.02, 5.8, 0.32, [("这些产品的共同特征", 14, NAVY, True)])
for i, txt in enumerate([
        "全球 1′ 覆盖，无海陆空缺",
        "已公开发布，可直接下载使用",
        "分辨率上限 1′（≈1.85 km）",
        "低频精度高，是高分辨率重建的基准"]):
    yy = 4.48 + i * 0.50
    textbox(s, 6.93, yy, 0.30, 0.26, [("·", 14, NAVY, True)])
    textbox(s, 7.20, yy - 0.02, 5.55, 0.30, [(txt, 11.5, DARK, False)])
strip(s, [N("全球 1′ 覆盖是它的优势，"), R("1′ 以下的短波在模型中缺失"),
          N(" 是它的天花板")])
notes(s, "测高反演的链条：海面高 → 沿轨求导得垂线偏差 → 逆 Vening-Meinesz 或谱方法 → "
         "重力异常。四个模型都是 1′，因为再细就没有测高信息支撑了。"
         "右侧四条问题逐条讲：病态反演、轨道间插值、单一数据源、1′ 采样率。")

# ============================================================ 4 主流方法（二）15″
s = new_slide("2.2", "主流方法（二）：15″ 高分辨率重建", "方法", [
    para(("要把分辨率推进到 ", 13.5, GREY, False),
         ("15″（≈460 m）", 13.5, RED, True),
         ("，必须把", 13.5, GREY, False),
         ("船测重力与海底地形", 13.5, RED, True),
         ("引入；主流方法有三条路线，各有各的代价。", 13.5, GREY, False)),
])
picture(s, "fig2_gap.png", 1.56, 10.60, 2.02)
routes = [
    ("深度学习反演", "Zhu 等, IEEE TGRS 61: 1–11, 2023",
     ["需要成对训练样本", "本区没有 15″ 真值标签，网络输出仍以 1′ 网格作标签"]),
    ("频域分频融合", "Chen 等, GJI 236(3): 1262–1274, 2024",
     ["截止波长必须人为给定", "本区实测随区域在 0.8–27 km 之间变，不是一个稳定的客观量"]),
    ("测高与船测协同融合", "刘乾坤 等, 空间科学学报, 2026",
     ["需给定融合权重", "权重随区域与水深变化，且产出仍为 1′"]),
]
for k, (name, cite, lim) in enumerate(routes):
    x = 0.42 + k * 4.18
    rect(s, x, 3.76, 3.98, 2.85, PALE)
    rect(s, x, 3.76, 3.98, 0.10, NAVY)
    textbox(s, x + 0.22, 3.94, 3.6, 0.32, [(name, 13.5, NAVY, True)])
    textbox(s, x + 0.22, 4.28, 3.6, 0.34, [(cite, 9.5, GREY, False)])
    textbox(s, x + 0.22, 4.70, 3.6, 0.26, [("局限", 10.5, RED, True)])
    textbox(s, x + 0.22, 4.96, 3.58, 0.26, [(lim[0], 11, RED, True)])
    textbox(s, x + 0.22, 5.26, 3.58, 1.10, [(lim[1], 10.5, GREY, False)], spacing=1.3)
strip(s, [N("三条路线都受制于各自的代价，"), R("15″ 仍是当前的前沿"),
          N("；而且各自只解决了一环")])
notes(s, "左图是同一片海域 1′ 与 15″ 的对比 —— 1′ 是台阶状，15″ 才有细节。"
         "三张卡片各写一条路线的方法、代表工作与具体局限："
         "深度学习缺 15″ 真值标签；频域融合的截止波长随区域剧烈变化；"
         "协同融合需要人工给权重且产出仍是 1′。")

# ============================================================ 5 存在的问题
s = new_slide("3.1", "现有方法存在的问题", "问题", [
    para(("把上面两条路线、五种做法放在一起看，它们的局限可以归纳成", 13.5, GREY, False),
         ("四个卡点", 13.5, RED, True),
         ("。", 13.5, GREY, False)),
])
picture(s, "fig_problems.png", 1.72, 12.20, 3.30)
textbox(s, 0.42, 5.20, 12.5, 0.96,
        [("前两个卡点决定「能不能把三类数据用上」，后两个决定「结果能不能被检验」。"
          "四个卡点都不是算法本身不够好，而是信息没有被组织起来。", 12.5, DARK, False)],
        spacing=1.3)
strip(s, [N("四个卡点归结起来是一件事："), R("信息没有被写进同一套方程")])
notes(s, "① 只用一类数据：来自 2.1 的测高路线与空间插值；"
         "② 权重靠人为给定：来自 2.2 的频域融合（截止波长）与协同融合（融合权重）；"
         "③ 只输出一张场图：五种做法都没有逐像元不确定度；"
         "④ 停在 1′：公开模型与三条重建路线都以 1′ 为产出。")

# ============================================================ 6 卡尔曼滤波能解决什么
s = new_slide("4.1", "卡尔曼滤波能解决什么", "解决", [
    para(("卡尔曼滤波把", 13.5, GREY, False),
         ("观测、误差、状态", 13.5, RED, True),
         ("写成一组递推方程 —— 四个卡点可以", 13.5, GREY, False),
         ("一一对上", 13.5, RED, True),
         ("。", 13.5, GREY, False)),
])
picture(s, "fig_mapping.png", 1.66, 12.10, 3.55)
textbox(s, 0.42, 5.42, 12.5, 0.62,
        [("对应关系不是比喻：每一行的右边都是方程里真实存在的量，"
          "换掉任何一个，对应的卡点就会重新出现。", 12.5, DARK, False)])
strip(s, [N("它解决的不是某一环，而是让"), R("四环同时成立")])
notes(s, "逐行讲：观测向量 z 容纳三类观测；观测噪声 R 与过程噪声 Q 决定权重，"
         "由卡尔曼增益自动算出；状态协方差 P 直接给出逐像元不确定度；"
         "状态定义在 15″，于是解自然落在 1′ 以下。")

# ============================================================ 7 什么是卡尔曼滤波
s = new_slide("5.1", "什么是卡尔曼滤波", "原理", [
    para(("一个", 13.5, GREY, False),
         ("「预测 — 加权校正」", 13.5, RED, True),
         ("的递推循环，每一步都带着", 13.5, GREY, False),
         ("不确定度", 13.5, RED, True),
         ("。", 13.5, GREY, False)),
])
picture(s, "fig_kf_cycle.png", 1.58, 9.90, 3.95)
for k, (t_, a_, b_) in enumerate([
        ("① 一句话理解", "边预测，边修正", "先按物理规律推一步，再用观测纠偏"),
        ("② 为什么能超分", "观测在 1′，状态在 15″", "两者之间由观测矩阵 H 建立映射"),
        ("③ 与一次性求解的区别", "不解大矩阵，逐像元递推", "1.84e7 维状态向量也可算")]):
    x = 0.85 + k * 4.03
    rect(s, x, 5.46, 3.62, 1.10, PALE)
    rect(s, x, 5.46, 0.075, 1.10, NAVY)
    textbox(s, x + 0.28, 5.54, 3.2, 0.34, [(t_, 14.5, NAVY, True)])
    textbox(s, x + 0.28, 5.86, 3.25, 0.60,
            [(a_, 12.5, MID, False), (b_, 11.5, GREY, False)], spacing=1.1)
strip(s, [N("谁更可信就多听谁的 —— "), R("权重由误差自动给出"), N("，不需要人为设定")])
notes(s, "只讲通用原理：预测步用空间相关模型外推、不确定度增大；"
         "校正步用观测修正、不确定度减小；两者之间的权重就是卡尔曼增益。")

# ============================================================ 8 本项目的思路
s = new_slide("6.1", "本项目的思路", "思路", [
    para(("把", 13.5, GREY, False),
         ("卫星测高重力、船载重力、海底地形", 13.5, RED, True),
         ("三类数据写进同一套方程，", 13.5, GREY, False),
         ("状态就是 15″ 重力场", 13.5, RED, True),
         ("。", 13.5, GREY, False)),
])
picture(s, "fig1_pipeline.png", 1.60, 11.90, 3.15)
textbox(s, 0.42, 5.10, 12.5, 0.90,
        [("卫星测高给低频框架、船测给局部高精度约束、海底地形给高频物理先验 —— "
          "三者的角色不同，但都被写成观测方程，", 12.5, DARK, False),
         ("按各自精度参与，而不是简单加权。", 12.5, NAVY, True)], spacing=1.3)
strip(s, [N("三类观测各按自身精度参与，输出"), R("带逐像元不确定度")])
notes(s, "与前一页的对照图对应：观测向量容纳三类数据，权重由 R/Q 决定，"
         "状态携带方差。这一页只讲「三类数据怎么进来」，细节参数在下一页。")

# ============================================================ 9 本项目的流程
s = new_slide("7.1", "本项目的卡尔曼滤波流程", "流程", [
    para(("状态有 ", 13.5, GREY, False),
         ("1.84×10⁷", 13.5, RED, True),
         (" 个未知数，卫星观测只有 ", 13.5, GREY, False),
         ("1.15×10⁶", 13.5, RED, True),
         (" 个 —— 观测远少于未知数，只能递推求解。", 13.5, GREY, False)),
])
picture(s, "fig_project_flow.png", 1.60, 12.30, 4.30)
textbox(s, 0.42, 5.94, 12.5, 0.60,
        [("先验用 SSV 块填充叠加地形高频；六轮预测—校正交替，卫星与船测分两步更新；"
          "最后做一次反向平滑抑制路径效应。", 12.5, DARK, False)], spacing=1.15)
strip(s, [N("六轮递推 + 反向平滑，"), R("参数全域冻结"), N("，三个验证区共用同一套")])
notes(s, "参数：α=0.5、σ=4 像元、R_sat=9、R_ship=1、Q=4 mGal²、六轮迭代 + RTS 平滑。"
         "协方差取逐像元方差（对角近似），这是 1.84e7 维状态下的可算性取舍。")

# ============================================================ 10 验证设计
s = new_slide("8.1", "验证设计：留出协议与评估口径", "验证", [
    para(("三个", 13.5, GREY, False),
         ("互不重叠", 13.5, RED, True),
         ("的验证区；留出单元是", 13.5, GREY, False),
         ("整条航次", 13.5, RED, True),
         ("而不是单个测点；参数在三个区域上完全冻结。", 13.5, GREY, False)),
])
picture(s, "fig_validation_design.png", 1.62, 12.25, 4.05)
strip(s, [N("整条航次留出、参数全域冻结、"), R("12.7 万个验证点全部未参与解算")])
notes(s, "验证设计是这份工作最容易被质疑的地方，所以要单独一页。"
         "三个区域互不重叠；20 条航次里 17 条参与解算、3 条整条留出；"
         "参数在三个区域上完全冻结，因此区域间差异只来自数据本身。")

# ============================================================ 11 结果（一）
s = new_slide("8.2", "结果（一）：三个区域的横向对比", "验证", [
    para(("15″ 点口径下，与", 13.5, GREY, False),
         ("已发布的 SSV33.1 模型", 13.5, RED, True),
         ("及三种插值基线、频域融合逐一对比。", 13.5, GREY, False)),
])
picture(s, "fig_results.png", 1.58, 12.10, 3.95)
textbox(s, 0.42, 5.62, 12.5, 0.72,
        [("插值类三种基线均由第三方库实现（Verde v1.9.0 的 KNeighbors / Linear、"
          "PyKrige v1.7.3 的 OrdinaryKriging），与本方法使用完全相同的留出航次与评估函数。",
          12, GREY, False)], spacing=1.15)
strip(s, [N("三个验证区、15 个对比格次，卡尔曼滤波全部取优，改善 "),
          R("9.4% ~ 10.7%")])
notes(s, "15″ 点口径下（mGal）：综合验证区 4.173（对 SSV 4.674）、南海-琉球 4.748、"
         "菲律宾海中部 1.455；三个区域都用同一套参数，未做任何重调。")

# ============================================================ 12 结果（二）
s = new_slide("8.3", "结果（二）：同一份结果，两种口径", "验证", [
    para(("同一份 15″ 结果，用", 13.5, GREY, False),
         ("两种口径", 13.5, RED, True),
         ("去评，差别可达 1.03 mGal —— 所以报精度", 13.5, GREY, False),
         ("必须写明口径", 13.5, RED, True),
         ("。", 13.5, GREY, False)),
])
picture(s, "fig6_scales.png", 1.66, 11.90, 3.75)
textbox(s, 0.42, 5.42, 12.5, 0.86,
        [("15″ 点口径：模型点直接对船载原始 10 s 采样点；", 12.5, DARK, False),
         ("1′ 块口径：模型块均值对船载块均值 —— 后者是国际上模型间对比的通用口径。",
          12.5, NAVY, True)], spacing=1.3)
strip(s, [N("两种口径差 "), R("0.02 ~ 1.03 mGal"),
          N("，报告中必须注明用的是哪一种")])
notes(s, "口径差异来自采样：15″ 点对点保留了船测的高频起伏，块均值则先平滑。"
         "国际上模型间对比多用 1′ 块口径，所以两种都报。")

# ============================================================ 13 结果（三）
s = new_slide("8.4", "结果（三）：改善幅度分解", "验证", [
    para(("把改善量", 13.5, GREY, False),
         ("按基线拆开", 13.5, RED, True),
         ("看：相对卫星模型与相对频域融合分别是多少。", 13.5, GREY, False)),
])
picture(s, "fig_improvement.png", 1.66, 11.90, 3.90)
textbox(s, 0.42, 5.46, 12.5, 0.86,
        [("相对卫星模型的改善（0.15–0.50 mGal）明显大于相对频域融合（0.11–0.40 mGal）—— "
          "说明频域融合本身已经吃掉了一部分短波，本方法在它之上仍有增益。",
          12.5, DARK, False)], spacing=1.3)
strip(s, [N("相对卫星模型 "), R("9.4% ~ 10.7%"),
          N("，相对频域融合 "), R("6.9% ~ 8.8%"), N("，两者都为正")])
notes(s, "这张图回答「改善是不是只靠换基线」：两条基线都对比了，都是正的。"
         "菲律宾海中部绝对量小是因为该区本身误差水平就低（1.5 mGal 量级）。")

# ============================================================ 14 总结
s = new_slide("9.1", "总结", "总结", [
    para(("三类观测递推求解 15″ 重力场，", 13.5, GREY, False),
         ("三个验证区改善 9.4% ~ 10.7%", 13.5, RED, True),
         ("。", 13.5, GREY, False)),
])
cards = [
    ("以 15″ 重力场为状态", "把卫星测高重力、船载重力、海底地形三类观测写进同一套方程，"
                        "状态就是待求的 15″ 重力场", NAVY),
    ("把观测写成同一套方程", "五种现有做法各缺一环；只有把观测写成方程，"
                       "多源同解、权重自定、精度可查才能同时成立", MID),
    ("三个区改善 9.4%~10.7%", "12.7 万个验证点、15 个对比格次全部取优，"
                             "且三个区域共用同一套参数", RED),
]
for k, (h_, line, c) in enumerate(cards):
    x = 0.85 + k * 4.06
    rect(s, x, 1.86, 3.66, 3.10, PALE)
    rect(s, x, 1.86, 3.66, 0.11, c)
    textbox(s, x + 0.28, 2.16, 3.16, 0.52, [(h_, 17, c, True)])
    textbox(s, x + 0.30, 2.76, 3.14, 2.00, [(line, 13, DARK, False)], spacing=1.35)
textbox(s, 0.85, 5.26, 12.0, 1.05,
        [("下一步：", 13, NAVY, True),
         ("用公开的俯冲带剩余重力格网独立复验；把逐像元方差用于标定可信区间与排序补测航段。",
          12.5, GREY, False)], spacing=1.3)
strip(s, [N("三类观测同解、权重自定、精度可查 —— "), R("这是相对其余五类方法的立足点")])
notes(s, "总结页在提问环节一直挂在屏幕上，三张卡片对应三个问题。")

prs.save(str(OUT))
print("已保存:", OUT)
print("页数:", len(prs.slides._sldIdLst))
