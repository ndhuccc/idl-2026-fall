"""Fixed, declared teaching controls and unchanged source examples for Ch01 KP05–07.
Run with Python; PNG lecture views and dark SVG slide views share data and geometry.
"""
from pathlib import Path
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Polygon
from playwright.sync_api import sync_playwright
OUT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'svg.fonttype':'path','font.family':'DejaVu Sans','font.size':18,'axes.labelsize':19,'xtick.labelsize':16,'ytick.labelsize':16})
MAIN,REF,EDGE,AUX='#d97757','#6a9fd6','#e0995f','#5fb8a8'
X=np.array([[-3,-1.2],[3,-1.2],[-3,-.8],[3,-.8],[-3,.8],[3,.8],[-3,1.2],[3,1.2]])
labels=np.array([0]*4+[1]*4)
COLORS=[REF,MAIN]
def save(fig,name,dark):
    if dark:fig.savefig(OUT/(name+'.svg'),facecolor=fig.get_facecolor())
    else:fig.savefig(OUT/(name+'.png'),dpi=200,facecolor=fig.get_facecolor())
    plt.close(fig)
def axes(ax,bg,fg):
    ax.set_facecolor(bg);ax.tick_params(colors=fg)
    for s in ax.spines.values():s.set_color(fg)
    ax.xaxis.label.set_color(fg);ax.yaxis.label.set_color(fg);ax.grid(alpha=.18,color=fg)
def normal(x,mu,sd):return np.exp(-.5*((x-mu)/sd)**2)/(sd*np.sqrt(2*np.pi))
def ellipse(ax,cov,color,label,fg):
    vals,U=np.linalg.eigh(cov);angle=np.degrees(np.arctan2(U[1,1],U[0,1]))
    ax.add_patch(Ellipse((0,0),2*np.sqrt(vals[1]),2*np.sqrt(vals[0]),angle=angle,fill=False,edgecolor=color,lw=3,label=label))

for dark in [False,True]:
    bg,fg=('#151513','#f2ede4') if dark else ('white','#172a36')
    fig=plt.figure(figsize=(11.6,3.65),facecolor=bg)
    a=fig.add_axes([.065,.21,.31,.56]);axes(a,bg,fg);a.set_xlim(-4,4);a.set_ylim(-1.7,1.7);a.set_aspect('equal')
    for k in [0,1]:a.scatter(X[labels==k,0],X[labels==k,1],color=COLORS[k],s=65,edgecolors=fg,linewidths=.5)
    a.set_xlabel('x1');a.set_ylabel('x2');a.set_title('Same 8 points; labels 1 / 2',color=fg,fontsize=18)
    a.annotate('',(2,0),(0,0),arrowprops={'arrowstyle':'->','color':AUX,'lw':3})
    a.annotate('',(0,1.5),(0,0),arrowprops={'arrowstyle':'->','color':EDGE,'lw':3})
    for j,(coordinate,title) in enumerate([(0,'PCA: retain x1 (variance 9)'),(1,'FDA: retain x2 (class separation)')]):
        ax=fig.add_axes([.49,.57-j*.39,.47,.23]);axes(ax,bg,fg);ax.set_ylim(-.3,.35);ax.set_yticks([]);ax.set_xlim((-4,4) if coordinate==0 else (-1.7,1.7));ax.axhline(0,color=fg,lw=1)
        for k in [0,1]:
            vals,counts=np.unique(X[labels==k,coordinate],return_counts=True)
            dy=(.13 if k==0 else -.13) if coordinate==0 else 0
            ax.scatter(vals,np.full(len(vals),dy),color=COLORS[k],s=60,edgecolors=fg,linewidths=.5)
            for v,n in zip(vals,counts):ax.text(v,dy+(.06 if coordinate==1 or k==0 else -.08),'×'+str(n),ha='center',va='top' if coordinate==0 and k==1 else 'bottom',color=fg,fontsize=15)
        ax.set_title(title,color=fg,fontsize=18)
    fig.text(.5,.035,'PCA display offsets show classes only; all 8 samples remain. Added fixed example.',color=fg,ha='center',fontsize=16)
    save(fig,'lec01-kp05-projection-criteria',dark)

    fig,aa=plt.subplots(1,2,figsize=(11.6,3.5),facecolor=bg)
    for ax,vals,title in zip(aa,[[1,2],[4,-3]],['Immediate reward: step 1','Two-step return: r1 + r2']):
        axes(ax,bg,fg);ax.bar([0,1],vals,color=[REF,MAIN],width=.55);ax.set_xticks([0,1],['A','B']);ax.set_ylim(-5.2,6);ax.axhline(0,color=fg,lw=1);ax.set_title(title,color=fg,fontsize=20)
        for i,v in enumerate(vals):ax.text(i,v+(.22 if v>=0 else -.2),f'{v:+g}',ha='center',va='bottom' if v>=0 else 'top',color=fg,fontsize=22)
    aa[0].set_ylabel('Reward units');fig.subplots_adjust(left=.08,right=.98,bottom=.19,top=.79,wspace=.3)
    save(fig,'lec01-kp05-reward-sequences',dark)

    fig,aa=plt.subplots(1,2,figsize=(11.6,4),facecolor=bg)
    xx=np.linspace(-.15,1.15,600);fq=[normal(xx,.8,.1),normal(xx,.2,.1)]
    for ax,scale,title in zip(aa,[1,.5],['Class densities f1, f2','Prior-weighted densities q1, q2']):
        axes(ax,bg,fg)
        for k in [0,1]:ax.plot(xx,scale*fq[k],color=COLORS[k],lw=3,label='Class '+str(k+1))
        ax.axvline(.7,color=EDGE,lw=2,ls='--');ax.set_xlim(-.15,1.15);ax.set_ylim(0,4.6*scale);ax.set_xlabel('Roundness x');ax.set_title(title,color=fg,fontsize=18)
        ax.text(.02,.91,'x=0.7',transform=ax.transAxes,color=fg,fontsize=18)
    aa[0].set_ylabel('Density (not point probability)');aa[1].legend(loc='upper right',frameon=False,labelcolor=fg,fontsize=16)
    fig.subplots_adjust(left=.08,right=.98,bottom=.22,top=.83,wspace=.34)
    save(fig,'lec01-kp06-bayes-roundness',dark)

    fig,aa=plt.subplots(1,2,figsize=(11.6,3.8),facecolor=bg)
    gx=np.linspace(-3,3,450);gy=np.linspace(-2,2,450);xx,yy=np.meshgrid(gx,gy);grid=np.stack([xx,yy],axis=-1)
    means=[np.array([-1,0]),np.array([1,0])]
    covs=[np.diag([1,.5]),np.diag([1,.5])];qcovs=[np.diag([.4,.8]),np.diag([1.8,.3])]
    for ax,Cs,title in zip(aa,[covs,qcovs],['LDA: shared covariance','QDA: separate covariances']):
        axes(ax,bg,fg);score=[]
        for m,C in zip(means,Cs):
            diff=grid-m;score.append(-.5*np.einsum('...i,ij,...j->...',diff,np.linalg.inv(C),diff)-.5*np.log(np.linalg.det(C)))
        s=score[0]-score[1];ax.contourf(xx,yy,s,levels=[s.min()-1,0,s.max()+1],colors=[MAIN,REF],alpha=.12);ax.contour(xx,yy,s,levels=[0],colors=[EDGE],linewidths=2.5)
        for k,(m,C) in enumerate(zip(means,Cs)):
            ax.scatter(*m,c=COLORS[k],s=70,edgecolors=fg);ax.text(m[0],m[1]-.35,'mu'+str(k+1),ha='center',color=fg,fontsize=16)
        ax.set_xlim(-3,3);ax.set_ylim(-2,2);ax.set_aspect('equal');ax.set_xlabel('x1');ax.set_title(title,color=fg,fontsize=20)
    aa[0].set_ylabel('x2');fig.subplots_adjust(left=.065,right=.985,bottom=.2,top=.83,wspace=.25)
    save(fig,'lec01-kp06-covariance-boundaries',dark)

    z=np.array([3,.5]);R=z/2;L=np.array([2,0]);rad=np.linalg.norm(R)
    fig,aa=plt.subplots(1,2,figsize=(11.6,3.45),facecolor=bg)
    ax=aa[0];axes(ax,bg,fg);ax.set_xlim(-2.3,3.8);ax.set_ylim(-2.1,2.1);ax.set_aspect('equal');ax.add_patch(plt.Circle((0,0),rad,fill=False,color=REF,lw=2.5));ax.add_patch(Polygon([[2,0],[0,2],[-2,0],[0,-2]],fill=False,color=MAIN,lw=2.5))
    for point,color in [(z,fg),(R,REF),(L,MAIN)]:ax.scatter(*point,color=color,s=70)
    ax.set_xlabel('beta1');ax.set_ylabel('beta2');ax.set_title('L2 circle; L1 corners',color=fg,fontsize=19)
    ax=aa[1];axes(ax,bg,fg);x=np.arange(3)
    ax.bar(x-.16,[3,1.5,2],.3,color=AUX,label='beta1');ax.bar(x+.16,[.5,.25,0],.3,color=EDGE,label='beta2');ax.set_xticks(x,['z','Ridge','Lasso']);ax.set_ylim(0,3.8);ax.set_title('Same z and lambda=1',color=fg,fontsize=19)
    for j,a in enumerate([[3,.5],[1.5,.25],[2,0]]):
        for offset,v in zip([-.16,.16],a):ax.text(j+offset,v+.08,str(v),ha='center',color=fg,fontsize=17)
    ax.legend(loc='upper right',frameon=False,labelcolor=fg,fontsize=16);fig.subplots_adjust(left=.075,right=.985,bottom=.22,top=.8,wspace=.28)
    save(fig,'lec01-kp07-penalty-geometry',dark)

    fig,aa=plt.subplots(1,2,figsize=(11.6,2.65),facecolor=bg)
    for ax,N,title in zip(aa,[np.diag([.5,1.04]),np.diag([1.04,1.04])],['FA noise: diag(0.5, 1.04)','pPCA noise: 1.04 I']):
        axes(ax,bg,fg);ax.set_xlim(-3.4,3.4);ax.set_ylim(-1.5,1.5);ax.set_aspect('equal');ellipse(ax,np.diag([9,1.04]),AUX,'Total covariance',fg);ellipse(ax,N,REF,'Noise',fg);ax.set_title(title,color=fg,fontsize=19);ax.set_xlabel('coordinate 1');ax.set_xticks([-3,0,3]);ax.set_yticks([-1,0,1])
    aa[0].set_ylabel('coordinate 2');fig.subplots_adjust(left=.08,right=.98,bottom=.25,top=.75,wspace=.28)
    save(fig,'lec01-kp07-noise-assumptions',dark)

def svg(body,h=440):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="{h}" viewBox="0 0 1280 {h}"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10Z" fill="#6a9fd6"/></marker></defs><rect width="1280" height="{h}" fill="white"/><g font-family="Arial" font-size="28" fill="#172a36">{body}</g></svg>'
def box(x,y,w,h,lines,color=REF):
    b=f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{color}" fill-opacity=".1" stroke="{color}" stroke-width="3"/>'
    for i,line in enumerate(lines):b+=f'<text x="{x+w/2}" y="{y+40+i*35}" text-anchor="middle">{line}</text>'
    return b
def line(x1,y1,x2,y2):return f'<path d="M{x1} {y1}L{x2} {y2}" fill="none" stroke="{REF}" stroke-width="4" marker-end="url(#arrow)"/>'
native={}
body=box(60,120,360,180,['Agent','state s → action a'],MAIN)+box(860,120,360,180,['Environment','next state s\' and reward r'])+line(420,160,860,160)+line(860,260,420,260)
body+='<text x="640" y="139" text-anchor="middle">action a</text><text x="640" y="313" text-anchor="middle">next state s\' / reward r</text><circle cx="570" cy="355" r="27" fill="#7dbb8c"/><text x="570" y="364" text-anchor="middle">+</text><circle cx="710" cy="355" r="27" fill="#e07a70"/><text x="710" y="364" text-anchor="middle">−</text><text x="640" y="414" text-anchor="middle">Repeat interaction; a single reward does not prove learning.</text>'
native['lec01-kp05-agent-environment-loop']=svg(body)
body=box(450,20,380,85,['Supervised classification'],MAIN)
names=[['Bayes','LDA / QDA'],['Logistic','probabilities'],['KNN','metric','learning'],['Trees','random forest'],['SVM','kernel SVM'],['Ensembles','boosting']]
for j,lines in enumerate(names):
    x=20+j*210;body+=line(640,105,x+95,215)+box(x,215,190,115,lines)
body+='<text x="640" y="395" text-anchor="middle">Overlapping families; a method map, not a historical derivation.</text>'
native['lec01-kp06-classifier-families']=svg(body)
body=box(40,95,230,145,['Latent z','1D'],MAIN)+box(430,95,420,145,['Signal Wz + mean','2D'],AUX)+box(1010,95,230,145,['Observed x','2D'])+line(270,165,430,165)+line(850,165,1010,165)+box(485,310,310,90,['Noise epsilon'],EDGE)+line(795,355,1070,240)
body+='<text x="640" y="440" text-anchor="middle">Same total covariance can have different signal/noise decompositions.</text>'
native['lec01-kp07-latent-generation']=svg(body,475)
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    for name,content in native.items():
        path=OUT/(name+'.svg');path.write_text(content,encoding='utf-8')
        tab=browser.new_page(viewport={'width':1280,'height':475 if name.endswith('generation') else 440},device_scale_factor=2)
        tab.set_content('<style>body{margin:0}</style>'+content);tab.screenshot(path=str(path.with_suffix('.png')));tab.close()
    browser.close()
print('Generated 9 PNG/SVG pairs: 6 quantitative figures and 3 native-structure lecture views.')
