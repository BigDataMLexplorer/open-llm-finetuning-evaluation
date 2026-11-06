import json
from pathlib import Path
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
p=json.loads((ROOT/'artifacts/benchmark.json').read_text())['summary']
labels=['Base','Full FT','LoRA','LoRA + replay']
formal=[p['base_formal_accuracy']['mean']*100,p['full_ft_formal_accuracy']['mean']*100,p['lora_formal_accuracy']['mean']*100,p['replay_formal_accuracy']['mean']*100]
col=[p['base_colloquial_accuracy']['mean']*100,p['full_ft_colloquial_accuracy']['mean']*100,p['lora_colloquial_accuracy']['mean']*100,p['replay_colloquial_accuracy']['mean']*100]
fig,ax=plt.subplots(figsize=(10.5,5.8));fig.patch.set_facecolor('#07101d');ax.set_facecolor('#091524')
x=range(len(labels));w=.36
b1=ax.bar([i-w/2 for i in x],formal,width=w,label='Original style retention',color='#6f82ff')
b2=ax.bar([i+w/2 for i in x],col,width=w,label='Adapted style accuracy',color='#59c7b8')
ax.set_ylim(0,105);ax.set_xticks(list(x),labels);ax.set_ylabel('Accuracy (%)',color='#a9bbd2');ax.set_title('PEFT adaptation: learn the new style without forgetting the original',color='#edf4ff',fontsize=17,fontweight='bold',pad=17)
ax.tick_params(axis='x',colors='#dce8f8');ax.tick_params(axis='y',colors='#8fa4bf');ax.yaxis.grid(True,color='#203753',alpha=.7);ax.set_axisbelow(True)
for s in ax.spines.values():s.set_color('#284262')
leg=ax.legend(frameon=False);[t.set_color('#c8d6e8') for t in leg.get_texts()]
for bars in [b1,b2]:
    for bar in bars:
        v=bar.get_height();ax.text(bar.get_x()+bar.get_width()/2,v+1,f'{v:.1f}',ha='center',color='#edf4ff',fontsize=9,fontweight='bold')
ax.text(.5,-.16,'3 deterministic seeds · synthetic intent paraphrase shift · measured CI benchmark',transform=ax.transAxes,ha='center',color='#7f96b4',fontsize=10)
fig.tight_layout();out=ROOT/'artifacts'/'retention_benchmark.png';fig.savefig(out,dpi=180,facecolor=fig.get_facecolor(),bbox_inches='tight');print(out)
