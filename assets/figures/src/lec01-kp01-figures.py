"""Reproduce KP01 diagrams and the source's A/B score comparison.
No test errors are invented. Concept diagrams are SVG; only the numeric plot uses matplotlib.
Run from any directory. Requires matplotlib and Playwright's Chromium.
"""
from pathlib import Path
from decimal import Decimal
import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'assets/figures'
OUT.mkdir(exist_ok=True)
A=(Decimal('0.5'),Decimal('0.1')*10)
B=(Decimal('1.0'),Decimal('0.1')*2)
assert sum(A)==Decimal('1.5') and sum(B)==Decimal('1.2')

def concept_svg(name, panels, arrows=False):
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="360" viewBox="0 0 1280 360">',
           '<rect width="1280" height="360" fill="white"/>',
           '<defs><marker id="end" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10" fill="#275f79"/></marker></defs>']
    for x,w,title,lines in panels:
        parts.append(f'<rect x="{x}" y="55" width="{w}" height="250" rx="16" fill="#f4f6f8" stroke="#275f79" stroke-width="3"/>')
        parts.append(f'<text x="{x+24}" y="105" font-family="Arial" font-size="28" font-weight="700" fill="#132634">{html.escape(title)}</text>')
        for j,line in enumerate(lines):
            parts.append(f'<text x="{x+24}" y="{158+j*46}" font-family="Arial" font-size="26" fill="#132634">{html.escape(line)}</text>')
    if arrows:
        for left,right in zip(panels,panels[1:]):
            x1=left[0]+left[1]+10;x2=right[0]-15
            parts.append(f'<path d="M{x1},180 L{x2},180" fill="none" stroke="#275f79" stroke-width="4" marker-end="url(#end)"/>')
    parts.append('</svg>')
    path=OUT/(name+'.svg');path.write_text('\n'.join(parts),encoding='utf-8')
    return path

flow=concept_svg('lec01-kp01-learning-workflow',[
    (60,300,'Hypothesis class',['Allowed candidates','Models A and B']),
    (450,300,'Scoring rule',['Error + penalty','Lower is better']),
    (840,380,'Search',['Evaluate candidates','Choose a minimizer'])],True)
generalization=concept_svg('lec01-kp01-generalization-check',[
    (60,560,'Known training evidence',['A: error 0.5; score 1.5','B: error 1.0; score 1.2','This rule selects B']),
    (660,560,'Unseen-data evidence',['Test error: not measured','No guarantee from score','Evaluate separately'])])

for dark in (False,True):
    bg='#151513' if dark else 'white';fg='#f2ede4' if dark else '#132634'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':19,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(5.0,3.35),facecolor=bg)
    ax.set_facecolor(bg)
    xs=[0,1];err=[float(A[0]),float(B[0])];pen=[float(A[1]),float(B[1])]
    ax.bar(xs,err,width=.52,color='#6a9fd6',label='Training error')
    ax.bar(xs,pen,bottom=err,width=.52,color='#d97757',label='Penalty')
    for x,e,p in zip(xs,err,pen):
        ax.text(x,e/2,f'{e:.1f}',ha='center',va='center',color='#132634',fontsize=19)
        ax.text(x,e+p/2,f'{p:.1f}',ha='center',va='center',color='#132634',fontsize=19)
        ax.text(x,e+p+.04,f'{e+p:.1f}',ha='center',color=fg,fontsize=21,fontweight='bold')
    ax.set_xticks(xs,['A','B']);ax.set_xlim(-.65,1.65);ax.set_ylim(0,2.12)
    ax.set_yticks([0,.5,1,1.5]);ax.set_ylabel('Score',color=fg,fontsize=19)
    ax.tick_params(colors=fg,labelsize=18)
    for k in ['top','right']:ax.spines[k].set_visible(False)
    for k in ['bottom','left']:ax.spines[k].set_color(fg)
    ax.legend(loc='upper center',frameon=False,labelcolor=fg,fontsize=14,ncol=2,handlelength=.8,columnspacing=.7)
    fig.subplots_adjust(left=.19,right=.98,bottom=.16,top=.99)
    if dark:fig.savefig(OUT/'lec01-kp01-candidate-scores.svg',facecolor=bg)
    else:fig.savefig(OUT/'lec01-kp01-candidate-scores.png',dpi=200,facecolor=bg)
    plt.close(fig)

with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    page=browser.new_page(viewport={'width':1280,'height':360},device_scale_factor=2)
    for path in [flow,generalization]:
        page.set_content('<style>body{margin:0}</style>'+path.read_text(encoding='utf-8'))
        page.screenshot(path=str(path.with_suffix('.png')))
    browser.close()
print('Generated 3 SVG/PNG pairs; score totals A=1.5, B=1.2; no test values fabricated.')
