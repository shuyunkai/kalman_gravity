import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
C=json.loads(Path('output_real/three_parts/newdeck/regions_coherence.json').read_text(encoding='utf-8'))
B=json.loads(Path('output_real/three_parts/newdeck/regions_b_ckf.json').read_text(encoding='utf-8'))
S=json.loads(Path('output_real/three_parts/newdeck/regions_sdust_ckf.json').read_text(encoding='utf-8'))
regs=[('D_Nankai','南海—琉球'),('H_Kuroshio','九州东侧'),('G_Japan','日本海沟')]
rows=[(n,C[k]['ssv'],C[k]['ckf'],B[k]['b_ckf'],S[k]['sdust_ssv']) for k,n in regs]
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':9})
fig,ax=plt.subplots(figsize=(9,4.3),dpi=180); x=np.arange(3); w=.2
ax.bar(x-1.5*w,[r[1] for r in rows],w,label='SSV 1′（双线性）',color='#B8C2CC')
ax.bar(x-.5*w,[r[2] for r in rows],w,label='C：SSV 先验+块观测',color='#7F8C9A')
ax.bar(x+.5*w,[r[3] for r in rows],w,label='B：SSV 只作块观测',color='#C0392B')
ax.bar(x+1.5*w,[r[4] for r in rows],w,label='SDUST 先验+SSV 块观测',color='#0B2545')
for i,r in enumerate(rows):
 for j,v in enumerate(r[1:]): ax.text(i+(j-1.5)*w,v+.04,f'{v:.2f}',ha='center',fontsize=6.5,rotation=90)
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows]); ax.set_ylabel('留出航次 RMSE (mGal)')
ax.set_ylim(0,6.2); ax.grid(axis='y',alpha=.2); ax.legend(frameon=False,fontsize=7,ncol=2,loc='upper right')
ax.set_title('SSV 使用方式与独立先验的三区对比')
fig.tight_layout(); fig.savefig('output_real/three_parts/regions_C_B_SDUST.png',bbox_inches='tight')
Path('output_real/three_parts/sdust_summary.md').write_text('''# SDUST2022GRA 先验 + SSV 块观测（B+独立先验）

区域：南海—琉球、九州东侧、日本海沟
验证：YK14-14 / YK18-14 / MR22-04，完全留出
方法：SDUST2022GRA 1′ 长波先验 + 0.5×地形高频；SSV33.1 作 1′块观测；船测点观测；6轮+RTS
参数：P0=3 mGal，Q=4 mGal²，R_sat=9，R_ship=1

| 区域 | SSV 1′ | C | B | SDUST 先验+SSV 观测 |
|---|---:|---:|---:|---:|
| 南海—琉球 | 5.344 | 4.881 | 5.294 | **4.721** |
| 九州东侧 | 5.514 | 4.703 | 4.896 | **4.149** |
| 日本海沟 | 3.960 | 3.710 | 3.725 | **3.416** |

结论：三区均优于 C，分别改善 0.16、0.55、0.29 mGal；SSV 只作一次块观测，避免重复使用。
''',encoding='utf-8')
print(rows)
