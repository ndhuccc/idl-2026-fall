"""Draw lecture diagrams from the exact editable nodes and edges in the deck.

Positions are schematic and carry no numerical data. Run after saving the deck.
"""
from pathlib import Path
import json, math, re
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).parents[3]
FIG=Path(__file__).parents[1]
DOC=json.loads((ROOT/'slides/lec03-backprop-optimization-v2.jbd.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.family':['Microsoft JhengHei','DejaVu Sans'],'font.size':17,
                     'mathtext.fontset':'dejavusans','savefig.facecolor':'white'})

def hook(node,index,offset):
    t=node['transform'];w,h=node['width'],node['height']
    angle=index*math.pi/8;dx,dy=math.sin(angle),-math.cos(angle)
    dist=min(w/2/abs(dx) if abs(dx)>1e-6 else 1e9,h/2/abs(dy) if abs(dy)>1e-6 else 1e9)
    return t['x']-offset+w/2+dist*dx,t['y']+h/2+dist*dy

for fid in ('lec03-bp-v2-kp06-network','lec03-bp-v2-kp06-neuron','lec03-bp-v2-kp08-batches'):
    first=next(n for n in DOC['nodes'].values() if n['name']=='fig:'+fid)
    prefix=first['id'].rsplit('-n',1)[0]
    shapes=[n for n in DOC['nodes'].values() if n['id'].startswith(prefix) and n['type']=='shape']
    edges=[n for n in DOC['nodes'].values() if n['id'].startswith(prefix) and n['type']=='connector']
    offset=min(n['transform']['x'] for n in shapes)-40
    xmax=max(n['transform']['x']+n['width']-offset for n in shapes)+40
    ymax=max(n['transform']['y']+n['height'] for n in shapes)+160
    ymin=min(n['transform']['y'] for n in shapes)-140
    fig,ax=plt.subplots(figsize=(16,5.5),layout='constrained')
    ax.set(xlim=(0,xmax),ylim=(ymax,ymin));ax.axis('off')
    for n in shapes:
        x,y=n['transform']['x']-offset,n['transform']['y'];w,h=n['width'],n['height']
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=12',facecolor='#f4f7fb',edgecolor='#6a9fd6',linewidth=2))
        text=n['text'].replace('*','')
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',linespacing=1.6,color='#1c1712')
    for n in edges:
        c=n['connector'];a=DOC['nodes'][c['sourceNodeId']];b=DOC['nodes'][c['targetNodeId']]
        p1,p2=hook(a,c['sourceHook'],offset),hook(b,c['targetHook'],offset)
        arc=(-.45 if c['sourceHook']==8 else .45) if c.get('curve') else 0
        ax.add_patch(FancyArrowPatch(p1,p2,arrowstyle='-|>',mutation_scale=20,linewidth=2.2,color='#507cb0',connectionstyle=f'arc3,rad={arc}'))
        if c.get('label'):
            text=c['label']['text'];x=(p1[0]+p2[0])/2
            y=(p1[1]+p2[1])/2-25
            if c.get('curve'):y+=-80 if c['sourceHook']==0 else 80
            ax.text(x,y,text,ha='center',va='center',color='#1c1712',bbox={'facecolor':'white','edgecolor':'none','pad':3})
    fig.savefig(FIG/(fid+'.png'),dpi=180)
    plt.close(fig)
    print(fid,len(shapes),'editable nodes;',len(edges),'edges')
