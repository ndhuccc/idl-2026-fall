from pathlib import Path
import sys,numpy as np,base64
REPO=Path(__file__).parents[3]
FIG=Path(__file__).parents[1]
sys.path.insert(0,str(REPO/'.agents/skills/view-from-text'))
from jbd_figure import use_jbd_style,new_figure,svg_data_uri
import matplotlib.pyplot as plt

def trajectory(alpha):
    w=np.array([2.,1.]);v=np.zeros(2);a=[w.copy()]
    for k in range(40):
        v=alpha*v-.06*np.array([w[0],25*w[1]])
        w=w+v;a.append(w.copy())
    return np.array(a)
gd=trajectory(0);mom=trajectory(.8)
def draw(kind,dark):
    use_jbd_style(22)
    if not dark:
        plt.rcParams.update({'axes.facecolor':'white','figure.facecolor':'white','savefig.transparent':False,'text.color':'#1c1712','axes.labelcolor':'#1c1712','axes.edgecolor':'#555555','xtick.color':'#555555','ytick.color':'#555555'})
    fig,ax=new_figure(760,440,(96,76,30,55))
    if kind=='ravine':
        x=np.linspace(-.2,2.2,401);y=np.linspace(-1.2,1.2,401);X,Y=np.meshgrid(x,y)
        ax.contour(X,Y,.5*(X**2+25*Y**2),levels=[.1,.5,1,2,4,8,12],colors='#b7b0a5',linewidths=.7)
        ax.plot(gd[:,0],gd[:,1],'.-',c='#d97757',label='GD (40)')
        ax.plot(mom[:,0],mom[:,1],'.-',c='#6a9fd6',label='Momentum (40)')
        ax.set(xlim=(-.2,2.2),ylim=(-1.2,1.2),xlabel='x',ylabel='y');ax.set_aspect('equal',adjustable='box')
        ax.legend(loc='upper center',bbox_to_anchor=(.5,1.19),ncol=2,fontsize=12,framealpha=.1)
    elif kind=='vectors':
        v=np.array([.05,-.6]);h=np.array([.9,.36]);s=v+h
        for vv,c,l in [(v,'#d97757','Current'),(h,'#6a9fd6','Retained'),(s,'#5fb8a8','Sum')]:
            ax.quiver(0,0,*vv,angles='xy',scale_units='xy',scale=1,color=c,label=l)
        ax.plot([v[0],s[0]],[v[1],s[1]],'--',c='#6a9fd6')
        ax.set(xlim=(-.15,1.15),ylim=(-.75,.5),xlabel='x displacement',ylabel='y displacement');ax.set_aspect('equal',adjustable='box')
        ax.legend(loc='upper center',bbox_to_anchor=(.5,1.19),ncol=3,fontsize=11,framealpha=.1)
    elif kind=='square':
        x=np.linspace(.5,1.2,401);loss=.5*x**4
        ax.plot(x,loss,c='#d97757',label=r'$e(w)=\frac{1}{2}w^4$')
        ax.plot(x,.5+2*(x-1),c='#6a9fd6',label='Tangent at 1')
        ax.scatter([1,1.01],[.5,.5*1.01**4],c='#d97757')
        ax.set(xlim=(.5,1.2),ylim=(-.1,1.1),xlabel='Hidden weight w',ylabel='Loss e')
        ax.legend(loc='upper center',bbox_to_anchor=(.5,1.19),ncol=2,fontsize=11,framealpha=.1)
    elif kind=='gradients':
        x=np.arange(1,5);ax.bar(x,[-2,-1,-8,-5],color='#d97757',label='Demo sample gradient')
        ax.axhline(-4,c='#6a9fd6',ls='--',label='Full mean = -4')
        ax.set(xlim=(.3,4.7),ylim=(-9,1),xlabel='Sample index (n=4)',ylabel='Gradient at w=0',xticks=x)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,1.19),ncol=2,fontsize=11,framealpha=.1)
    elif kind=='schedule':
        x=np.linspace(1,16,401)
        for y,c,l in [(1/x,'#d97757','1/k'),(1/np.sqrt(x),'#6a9fd6','1/sqrt(k)'),(.2*np.ones_like(x),'#e0995f','0.2')]:ax.plot(x,y,c=c,label=l)
        ax.set(xlim=(1,16),ylim=(0,1.1),xlabel='Iteration k',ylabel='Learning rate')
        ax.legend(loc='upper center',bbox_to_anchor=(.5,1.19),ncol=3,fontsize=12,framealpha=.1)
    elif kind=='variance':
        x=np.array([1,2,4]);ax.plot(x,[5,5/3,0],'o-',c='#d97757',label='No replacement')
        ax.plot(x,[5,2.5,1.25],'s-',c='#6a9fd6',label='Replacement')
        ax.set(xlim=(.8,4.2),ylim=(-.3,5.5),xlabel='Batch size b (n=4)',ylabel='Gradient variance',xticks=x)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,1.19),ncol=2,fontsize=11,framealpha=.1)
    ax.grid(alpha=.15)
    return fig

if __name__=='__main__':
    fid='lec03-bp-v2-kp08-variance'
    fig=draw('variance',False);fig.savefig(FIG/(fid+'.png'),dpi=200);plt.close(fig)
    uri=svg_data_uri(draw('variance',True))
    (FIG/(fid+'.svg')).write_bytes(base64.b64decode(uri.split(',',1)[1]))
