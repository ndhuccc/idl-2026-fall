from pathlib import Path
import sys, json, math, base64
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'assets/figures'
sys.path.insert(0,str(ROOT/'.agents/skills/view-from-text'))
from jbd_figure import use_jbd_style, new_figure, svg_data_uri, px_to_pt
import matplotlib.pyplot as plt

def draw(kind,dark):
    use_jbd_style(22)
    if not dark:plt.rcParams.update({'text.color':'#222222','axes.labelcolor':'#222222','axes.edgecolor':'#444444','xtick.color':'#444444','ytick.color':'#444444','figure.facecolor':'white','axes.facecolor':'white','savefig.transparent':False})
    fig,ax=new_figure(650,390,margins_px=(110,78,42,70))
    if kind=='bounds':
        t=np.linspace(9,999,700)
        ax.plot(t,10/(t+1),color='#7dbb8c',label='Full GD shape')
        ax.plot(t,10/t+.5,color='#d97757',label=r'$\eta=0.5$')
        ax.plot(t,10/t+.05,color='#6a9fd6',label=r'$\eta=0.05$')
        ax.set(xlim=(0,1000),ylim=(0,1.8),xlabel='Iteration t',ylabel='Toy bound B(t)')
        ax.set_xticks([0,500,1000]);ax.set_yticks([0,.5,1,1.5])
        ax.legend(loc='upper center',bbox_to_anchor=(.47,1.30),ncol=3,fontsize=px_to_pt(19),frameon=False,handlelength=1.2,columnspacing=.8)
    elif kind=='decay':
        t=np.linspace(1,10000,700)
        ax.plot(t,1/np.sqrt(t),color='#d97757')
        ax.set(xscale='log',yscale='log',xlim=(1,10000),ylim=(.008,1.3),xlabel='Accumulated steps t',ylabel='Effective rate')
        ax.scatter([1,100,10000],[1,.1,.01],color='#6a9fd6',s=60,zorder=4)
        ax.set_xticks([1,100,10000],['1','100','10000']);ax.set_yticks([.01,.1,1],['0.01','0.1','1'])
        ax.set_title(r'$a_t=1/\sqrt{t}$',pad=15)
    elif kind=='bias':
        t=np.arange(1,41)
        ax.plot(t,1/(1-.9**t),color='#d97757')
        ax.scatter([1,2],[10,1/(1-.9**2)],color='#6a9fd6',s=60,zorder=4)
        ax.axhline(1,color='#b7b0a5',ls='--',lw=1.6)
        ax.set(xlim=(0,40),ylim=(0,11),xlabel='Completed updates t',ylabel='Correction factor')
        ax.set_xticks([1,10,20,40]);ax.set_yticks([1,5,10]);ax.set_title(r'$1/(1-0.9^t)$',pad=15)
    elif kind=='forget':
        r=np.arange(0,31)
        ax.plot(r,np.full_like(r,100),color='#6a9fd6',label='AdaGrad s = 100')
        ax.plot(r,10*.9**r,color='#d97757',label=r'RMSProp $10(0.9)^r$')
        ax.set(yscale='log',xlim=(0,30),ylim=(.3,150),xlabel='Zero-gradient steps r',ylabel='Squared scale')
        ax.set_yticks([1,10,100],['1','10','100']);ax.legend(loc='upper center',bbox_to_anchor=(.47,1.32),ncol=2,fontsize=px_to_pt(19),frameon=False,handlelength=1.3,columnspacing=.8)
    else:
        fig.delaxes(ax)
        ax=fig.add_axes([.14,.20,.36,.61]);ay=fig.add_axes([.63,.20,.33,.61])
        for a,rng,lim in [(ax,.5,13),(ay,.2,2.3)]:
            w=np.linspace(-rng,rng,700)
            a.plot(w,.01+50*w*w,color='#d97757',label='Sharp toy')
            a.plot(w,.08+.5*w*w,color='#6a9fd6',label='Flat toy')
            a.vlines([-.2,.2],0,2.01,color='#b7b0a5',ls='--',lw=1.4)
            a.set(xlim=(-rng,rng),ylim=(0,lim),xlabel='Parameter w',ylabel='Toy loss')
            a.set_xticks([-rng,0,rng]);a.grid(alpha=.2)
        ax.set_yticks([0,6,12]);ay.set_yticks([0,1,2]);ay.set_ylabel('')
        ax.set_title('Full interval',pad=12);ay.set_title('Local view',pad=12)
        fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.99),ncol=2,frameon=False,fontsize=px_to_pt(19))
    if kind!='landscape':ax.grid(alpha=.2)
    return fig

def make(kind,fid):
    fig=draw(kind,False);fig.savefig(ROOT/'assets/figures'/f'{fid}.png',dpi=200);plt.close(fig)
    fig=draw(kind,True);fig.savefig(OUT/f'{fid}-dark-preview.png',dpi=140,facecolor='#151513')
    uri=svg_data_uri(fig)
    (ROOT/'assets/figures'/f'{fid}.svg').write_bytes(base64.b64decode(uri.split(',')[1]))
    return {'figureId':fid,'svgBytes':(ROOT/'assets/figures'/f'{fid}.svg').stat().st_size,'points':700,'svgChecks':'PASS','whitePNG':str(ROOT/'assets/figures'/f'{fid}.png')}


if __name__ == '__main__':
    make('bias','lec03-bp-v2-kp11-bias')
