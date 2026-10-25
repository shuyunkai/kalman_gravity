import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.sans-serif":["Microsoft YaHei"],"axes.unicode_minus":False})
fig,ax=plt.subplots(figsize=(13.6,4.0),dpi=220); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
def box(x,y,w,h,t,fs=8.5):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.012,rounding_size=0.02",fc="#EEF3F9",ec="#0B2545",lw=1))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,color="#0B2545")
def arr(x0,x1,y=0.5):
    ax.annotate("",xy=(x1,y),xytext=(x0,y),arrowprops=dict(arrowstyle="-|>",color="#5B6573",lw=1.2))
for lab,y in (("卫星 1′",0.70),("船测重力",0.45),("地形 GEBCO",0.20)):
    box(0.02,y,0.12,0.18,lab)
box(0.18,0.30,0.14,0.40,"结构引导 KF\n结构张量预测\n保脊线/压噪声")
box(0.36,0.36,0.12,0.28,"15″ 重力场\n1.85 km")
box(0.52,0.30,0.14,0.40,"5–20 km 带通\nTHG 脊线\n细化/过滤")
box(0.70,0.36,0.10,0.28,"线性体")
box(0.84,0.36,0.12,0.28,"GM9 断层")
for y in (0.79,0.54,0.29): arr(0.14,0.18,y)
arr(0.32,0.36); arr(0.48,0.52); arr(0.66,0.70); arr(0.80,0.84)
ax.text(0.5,0.94,"15″ 重力场重建与断裂识别流程",ha="center",fontsize=11,color="#0B2545",fontweight="bold")
fig.tight_layout(); fig.savefig(r"output_real/three_parts/newdeck/fig_new_flow_fault.png",bbox_inches="tight"); print("ok")
