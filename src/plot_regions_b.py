import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
ROOT=Path('.')
C=json.loads((ROOT/'output_real/three_parts/newdeck/regions_coherence.json').read_text(encoding='utf-8'))
B=json.loads((ROOT/'output_real/three_parts/newdeck/regions_b_ckf.json').read_text(encoding='utf-8'))
regs=[('D_Nankai','南海—琉球'),('H_Kuroshio','九州东侧'),('G_Japan','日本海沟')]
rows=[]
for k,n in regs:
 rows.append((n,C[k]['ssv'],C[k]['ckf'],B[k]['b_ckf']))
print('region SSV C B')
for r in rows: print(r)
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':9})
fig,ax=plt.subplots(figsize=(8.5,4.2),dpi=180)
x=np.arange(len(rows)); w=.26
ax.bar(x-w, [r[1] for r in rows], w, label='SSV 1′（双线性）', color='#B8C2CC')
ax.bar(x, [r[2] for r in rows], w, label='C：SSV 先验+块观测', color='#0B2545')
ax.bar(x+w, [r[3] for r in rows], w, label='B：SSV 只作块观测', color='#C0392B')
for i,r in enumerate(rows):
 ax.text(i-w,r[1]+.03,f'{r[1]:.2f}',ha='center',fontsize=7)
 ax.text(i,r[2]+.03,f'{r[2]:.2f}',ha='center',fontsize=7)
 ax.text(i+w,r[3]+.03,f'{r[3]:.2f}',ha='center',fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows]); ax.set_ylabel('留出航次 RMSE (mGal)')
ax.set_ylim(0,6.1); ax.grid(axis='y',alpha=.2); ax.legend(frameon=False,fontsize=8,loc='upper right')
ax.set_title('B 与 C 的三区留出验证对比')
fig.tight_layout(); fig.savefig('output_real/three_parts/regions_b_vs_c.png',bbox_inches='tight')
