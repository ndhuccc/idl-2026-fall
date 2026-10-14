"""Reproduce Chapter 1 KP02–04 teaching graphics from source coordinates.
Conceptual diagrams use SVG. Geometric/function plots use analytic coordinates.
Source read 2026-10-14; b=-5 and retaining x1 are explicit added controls.
"""
from pathlib import Path
import html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'assets/figures'
P=np.array([[1.5,2],[2.5,2.5],[2,3.5],[3.5,3],[1,4],[7,6.5],[8,7.5],[7.5,5.5],[9,7],[8,8],[4.5,5],[5.5,4.5]])
G=np.array([[1,2],[1,3],[2,2],[4,2],[4,3],[5,2],[5.5,2],[5.5,3],[6.5,2],[10,2],[10,3],[11,2],[13,2],[13,3],[14,2]])
SEEDS={2:[[1,2.5],[13,2.5]],3:[[1.33,2.33],[5.83,2.33],[13.33,2.33]],5:[[1.33,2.33],[4.33,2.33],[5.83,2.33],[10.33,2.33],[13.33,2.33]]}
COLORS=['#d97757','#6a9fd6','#e0995f','#5fb8a8','#b7b0a5'];MARKERS=['o','s','^','D','P']

def label(x,y,t,size=28,bold=False):
    return f'<text x="{x}" y="{y}" font-family="Arial" font-size="{size}" font-weight="{700 if bold else 400}" fill="#132634">{html.escape(t)}</text>'
def svg(name,body,height=360):
    s=f'<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="{height}" viewBox="0 0 1280 {height}"><rect width="1280" height="{height}" fill="white"/><defs><marker id="end" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10" fill="#275f79"/></marker></defs>'+''.join(body)+'</svg>'
    path=OUT/(name+'.svg');path.write_text(s,encoding='utf-8');return path
body=[label(60,45,'Features are rows; samples are columns (d=2, n=12)',30,True)]
values=[['1.5','2.5','2.0','...'],['2.0','2.5','3.5','...']]
for j,t in enumerate(['sample 1','sample 2','sample 3','9 more']):body.append(label(210+j*245,103,t))
for i in range(2):
    body.append(label(35,182+i*85,f'feature {i+1}',24))
    for j in range(4):
        x=180+j*245;y=127+i*85
        body.append(f'<rect x="{x}" y="{y}" width="218" height="70" rx="10" fill="#edf3f7" stroke="#275f79" stroke-width="2"/>')
        body.append(label(x+72,y+47,values[i][j],34))
body.append(label(180,330,'One column is one 2D sample; the ellipsis is not another sample.',26))
matrix=svg('lec01-kp02-data-matrix',body)
body=[]
for x,w,title,lines in [(60,300,'Input',['x: 2 coordinates','d = 2']), (450,300,'Model',['theta: 3 parameters','2 weights + offset']), (840,380,'Output',['f_theta(x): 1 value','p = 1'])]:
    body.append(f'<rect x="{x}" y="65" width="{w}" height="220" rx="16" fill="#edf3f7" stroke="#275f79" stroke-width="3"/>')
    body.append(label(x+22,112,title,30,True))
    for j,t in enumerate(lines):body.append(label(x+22,177+j*48,t,25))
for x1,x2 in [(372,432),(762,822)]:body.append(f'<path d="M{x1},176 L{x2},176" fill="none" stroke="#275f79" stroke-width="4" marker-end="url(#end)"/>')
mapping=svg('lec01-kp02-model-mapping',body)
body=[]
for x,title,lines in [(60,'Classification',['Which category?','A finite class identifier','Example: apple or cucumber']), (660,'Regression',['What diameter?','A real-valued measurement','Example: 7.2 cm'])]:
    body.append(f'<rect x="{x}" y="50" width="560" height="260" rx="16" fill="#edf3f7" stroke="#275f79" stroke-width="3"/>')
    body.append(label(x+24,102,title,30,True))
    for j,t in enumerate(lines):body.append(label(x+24,164+j*48,t,26))
types=svg('lec01-kp03-output-types',body)

def skin(dark):
    bg='#151513' if dark else 'white';fg='#f2ede4' if dark else '#132634'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':20,'svg.fonttype':'none','mathtext.fontset':'cm'})
    return bg,fg
def axes(ax,bg,fg):
    ax.set_facecolor(bg);ax.tick_params(colors=fg,labelsize=18)
    for spine in ax.spines.values():spine.set_color(fg)
    ax.xaxis.label.set_color(fg);ax.yaxis.label.set_color(fg)
    ax.grid(alpha=.15,color=fg)
def save(fig,name,dark):
    fig.savefig(OUT/(name+('.svg' if dark else '.png')),dpi=200,facecolor=fig.get_facecolor());plt.close(fig)
def lloyd(m):
    c=np.array(SEEDS[m]);last=None
    for _ in range(100):
        ids=np.argmin(((G[:,None]-c[None])**2).sum(axis=2),axis=1)
        c=np.array([G[ids==k].mean(0) if np.any(ids==k) else c[k] for k in range(m)])
        if last is not None and np.array_equal(ids,last):break
        last=ids.copy()
    return ids,c

for dark in [False,True]:
    bg,fg=skin(dark)
    for classification,name in [(False,'lec01-kp02-affine-outputs'),(True,'lec01-kp03-shared-score')]:
        val=P.sum(1)-(5 if classification else 0)
        fig,aa=plt.subplots(1,2,figsize=(11.6,4.25),facecolor=bg,gridspec_kw={'width_ratios':[1,1]})
        ax=aa[0];axes(ax,bg,fg);ax.set_xlim(0,10);ax.set_ylim(0,10);ax.set_aspect('equal')
        ax.set_xlabel(r'$x_1$');ax.set_ylabel(r'$x_2$');ax.set_title('Input: n=12, d=2',color=fg,fontsize=22)
        if classification:
            t=np.linspace(0,5,400);ax.plot(t,5-t,color='#e0995f',lw=3)
            ax.scatter(P[:,0],P[:,1],c=[COLORS[0] if v>=0 else COLORS[1] for v in val],s=105,edgecolors=fg,linewidths=.7)
            ax.text(.3,9,'Boundary: x1+x2=5',color=fg,fontsize=18)
            handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=COLORS[1],markeredgecolor=fg,label='1: s < 0'),
                     Line2D([0],[0],marker='o',color='none',markerfacecolor=COLORS[0],markeredgecolor=fg,label='2: s ≥ 0')]
            ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(0,.84),frameon=False,labelcolor=fg,fontsize=18,handlelength=.6,handletextpad=.4)
        else:ax.scatter(P[:,0],P[:,1],c=val,cmap='coolwarm',vmin=0,vmax=16,s=105,edgecolors=fg,linewidths=.7)
        ax.annotate('sample 1',P[0],xytext=(.4,.5),color=fg,fontsize=18,arrowprops={'arrowstyle':'->','color':fg})
        ax=aa[1];axes(ax,bg,fg);ax.set_xlim(-3 if classification else 0,18);ax.set_ylim(-.6,.75)
        ax.set_yticks([]);ax.set_xlabel('score s' if classification else 'output value');ax.set_title('Continuous output',color=fg,fontsize=22)
        ax.axhline(0,color=fg,lw=1.5)
        ax.scatter(val,np.zeros(12),c=val,cmap='coolwarm',vmin=-2 if classification else 0,vmax=16,s=105,edgecolors=fg,linewidths=.7)
        ax.annotate('sample 1: '+f'{val[0]:g}',(val[0],0),xytext=(val[0],.4),ha='center',color=fg,fontsize=18,arrowprops={'arrowstyle':'->','color':fg})
        ax.text(.05,.1,'n=12; equal values overlap',transform=ax.transAxes,color=fg,fontsize=18)
        fig.subplots_adjust(left=.07,right=.98,bottom=.17,top=.83,wspace=.34)
        save(fig,name,dark)

    fig,aa=plt.subplots(1,3,figsize=(12.6,2.85),facecolor=bg)
    for ax,m in zip(aa,[2,3,5]):
        ids,c=lloyd(m);axes(ax,bg,fg);ax.set_xlim(0,15);ax.set_ylim(.8,4.2);ax.set_aspect('equal')
        for k in range(m):
            ax.scatter(G[ids==k,0],G[ids==k,1],c=COLORS[k],marker=MARKERS[k],s=100,edgecolors=fg,linewidths=.5)
            ax.scatter(c[k,0],c[k,1],c=fg,marker='x',s=155,linewidths=2.5)
        sizes='/'.join(str(n) for n in np.bincount(ids,minlength=m))
        ax.set_title(f'm={m}: sizes {sizes}',color=fg,fontsize=18)
        ax.set_xlabel(r'$x_1$',fontsize=19);ax.set_xticks([0,5,10,15]);ax.set_yticks([2,3])
    aa[0].set_ylabel(r'$x_2$',fontsize=19)
    fig.suptitle('Same n=15 points; source seeds; x marks centers',color=fg,fontsize=20)
    fig.subplots_adjust(left=.05,right=.99,bottom=.17,top=.77,wspace=.25)
    save(fig,'lec01-kp04-cluster-granularity',dark)

    fig,aa=plt.subplots(1,2,figsize=(10.6,3.35),facecolor=bg)
    ax=aa[0];axes(ax,bg,fg);ax.set_xlim(0,15);ax.set_ylim(.8,4.2);ax.set_aspect('equal');ax.set_xlabel(r'$x_1$');ax.set_ylabel(r'$x_2$')
    ax.scatter(G[:,0],G[:,1],c='#6a9fd6',s=100,edgecolors=fg,linewidths=.5)
    ax.set_title('2D: (1,2) and (1,3) differ',color=fg,fontsize=18)
    ax=aa[1];axes(ax,bg,fg);ax.set_xlim(0,15);ax.set_ylim(-.5,.75);ax.set_yticks([]);ax.set_xlabel(r'$z=x_1$')
    ax.axhline(0,color=fg,lw=1.5);u,counts=np.unique(G[:,0],return_counts=True)
    ax.scatter(u,np.zeros(len(u)),c='#d97757',s=100,edgecolors=fg,linewidths=.5)
    for v,c in zip(u,counts):
        if c>1:ax.text(v,.12,'×2',ha='center',color=fg,fontsize=16)
    ax.set_title('1D: both become 1',color=fg,fontsize=18)
    ax.text(.05,.1,'15 samples; 10 positions',transform=ax.transAxes,color=fg,fontsize=16)
    fig.subplots_adjust(left=.08,right=.99,bottom=.22,top=.8,wspace=.34)
    save(fig,'lec01-kp04-projection-loss',dark)

with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    tab=browser.new_page(viewport={'width':1280,'height':360},device_scale_factor=2)
    for path in [matrix,mapping,types]:
        tab.set_content('<style>body{margin:0}</style>'+path.read_text(encoding='utf-8'));tab.screenshot(path=str(path.with_suffix('.png')))
    browser.close()
print('Generated 7 PNG/SVG pairs from fixed source coordinates; no training performance is fabricated.')
