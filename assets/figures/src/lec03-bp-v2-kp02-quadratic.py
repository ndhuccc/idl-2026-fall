from pathlib import Path
import sys,base64
import numpy as np
ROOT=Path(r'C:\Users\ccchiang\Projects\idl-2026-fall')
FIG=ROOT/'assets/figures'
sys.path.insert(0,str(ROOT/'.agents/skills/view-from-text'))
from jbd_figure import use_jbd_style,new_figure,svg_data_uri,plt,PALETTE
FIGURES={}
def draw(name,kind):
    use_jbd_style(22)
    fig,ax=new_figure(640,440,margins_px=(108,84,28,64))
    if kind=='memory':
        n=np.geomspace(100,1e6,500)
        ax.plot(n,8*n,color=PALETTE['negative'],label=r'$8n$ (gradient)')
        ax.plot(n,8*n*n,color=PALETTE['positive'],label=r'$8n^2$ (Hessian)')
        for v in (100,10000,1000000):ax.scatter([v,v],[8*v,8*v*v],s=30,color=[PALETTE['negative'],PALETTE['positive']])
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(60,1.8e6);ax.set_ylim(200,2e13)
        ax.set_xlabel('Parameters n');ax.set_ylabel('Bytes');ax.legend(loc='upper left',fontsize=15.84)
    elif kind in ('gd','bounce'):
        x=np.linspace(-4.7,4.7,600);ax.plot(x,x*x,color=PALETTE['negative'],label=r'$f(w)=w^2$')
        ws=[4,3.2] if kind=='gd' else [4,-4]
        ax.scatter(ws,np.square(ws),s=75,color=PALETTE['positive'])
        for w in ws:ax.annotate(f'({w:g}, {w*w:g})',(w,w*w),xytext=(-65,10) if w>0 else (8,10),textcoords='offset points',fontsize=15.84)
        ax.set_xlim(-5.1,5.1);ax.set_ylim(-1,23);ax.set_xlabel('w');ax.set_ylabel('Loss f(w)');ax.legend(loc='upper center',fontsize=15.84)
    elif kind=='stop':
        k=np.arange(4);w=.5**k;g=2*w;df=np.r_[np.nan,np.abs(np.diff(w*w))]
        ax.plot(k,g,'o-',color=PALETTE['negative'],label=r'$|f\prime(w_k)|$')
        ax.plot(k,df,'s-',color=PALETTE['positive'],label=r'$|f_k-f_{k-1}|$')
        ax.hlines(.05,0,3,color=PALETTE['boundary'],ls='--',label=r'$\epsilon=0.05$')
        ax.set_xlim(-.2,3.3);ax.set_ylim(-.04,2.25);ax.set_xticks(k);ax.set_xlabel('Iteration k');ax.set_ylabel('Indicator');ax.legend(loc='upper right',fontsize=15.84)
    elif kind=='saddle':
        t=np.linspace(-2,2,600);ax.plot(t,t*t,color=PALETTE['negative'],label=r'$f(x,0)=x^2$')
        ax.plot(t,-t*t,color=PALETTE['positive'],label=r'$f(0,y)=-y^2$');ax.scatter([0],[0],color=PALETTE['warning'],s=70)
        ax.set_xlim(-2.25,2.25);ax.set_ylim(-4.7,5.6);ax.set_xlabel('Section coordinate');ax.set_ylabel('Loss');ax.legend(loc='upper center',fontsize=15.84)
    elif kind=='line':
        eta=np.linspace(0,1.1,600);ax.plot(eta,(3-6*eta)**2,color=PALETTE['negative'],label=r'$\phi(\eta)=(3-6\eta)^2$')
        ax.hlines(9,0,1.1,color=PALETTE['boundary'],ls='--',label='Current loss = 9')
        ax.scatter([1,.5],[9,0],color=PALETTE['positive'],s=70)
        ax.annotate('eta=1: 9',(1,9),xytext=(-115,-54),textcoords='offset points',fontsize=15.84)
        ax.annotate('eta=0.5: 0',(.5,0),xytext=(-10,45),textcoords='offset points',fontsize=15.84)
        ax.set_xlim(-.07,1.16);ax.set_ylim(-.7,15.5);ax.set_xlabel(r'$\eta$');ax.set_ylabel('Candidate loss');ax.legend(loc='upper center',fontsize=15.84)
    ax.grid(alpha=.17)
    # save the white-background lecture PNG from the same artist/data graph.
    for obj in [ax.xaxis.label,ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels()]:obj.set_color('#222222')
    for spine in ax.spines.values():spine.set_color('#666666')
    leg=ax.get_legend()
    if leg:
        for t in leg.get_texts():t.set_color('#222222')
        leg.get_frame().set_facecolor('white');leg.get_frame().set_alpha(.9)
    for a in ax.texts:a.set_color('#222222')
    fig.savefig(FIG/(name+'.png'),dpi=160,facecolor='white',transparent=False)
    for obj in [ax.xaxis.label,ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels()]:obj.set_color(PALETTE['text'])
    for spine in ax.spines.values():spine.set_color(PALETTE['muted'])
    if leg:
        for t in leg.get_texts():t.set_color(PALETTE['text'])
        leg.get_frame().set_facecolor('#151513');leg.get_frame().set_alpha(.95)
    for a in ax.texts:a.set_color(PALETTE['text'])
    uri=svg_data_uri(fig)',a)
    drawsrc=drawsrc[:a]+drawsrc[b:]
    (FIG/'src').mkdir(exist_ok=True)
    (FIG/'src'/(name+'.py')).write_text(source+drawsrc+f'\ndraw({name!r},{kind!r})\n',encoding='utf-8')
    uri=svg_data_uri(fig)
    data=base64.b64decode(uri.split(',',1)[1])
    (FIG/(name+'.svg')).write_bytes(data)
    FIGURES[name]={'width':640,'height':440,'bytes':len(data),'kind':kind}

draw('lec03-bp-v2-kp02-quadratic','gd')
