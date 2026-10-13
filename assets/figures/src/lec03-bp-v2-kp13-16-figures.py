"""Reproducible source-example illustrations, not measured network results."""
from pathlib import Path
import sys, os, json
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'.agents/skills/view-from-text'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/bp-v2-mpl'))
from jbd_figure import use_jbd_style,new_figure,svg_data_uri,PALETTE
import matplotlib.pyplot as plt
OUT=ROOT/'assets/figures'

def numeric_values():
    g=np.array([3.,4.]); eps=.1*g/np.linalg.norm(g)
    mask=np.array([1,0,1,0]); four=np.array([.06,.08,.10,.12]); swp=mask*four/.5
    toy=[30000*np.log(1/e)/d**2 for e,d in [(.01,.2),(.0001,.2),(.01,.1)]]
    assert np.allclose(eps,[.06,.08]) and np.isclose(np.linalg.norm(eps),.1)
    assert np.allclose(swp,[.12,0,.20,0])
    return {'SAM':{'gradient':g.tolist(),'norm':float(np.linalg.norm(g)),'epsilon':eps.tolist(),
                   'epsilonNorm':float(np.linalg.norm(eps)),'neighbor':(np.array([1.,2.])+eps).tolist()},
            'SWP':{'mask':mask.tolist(),'beta':.5,'perturbation':swp.tolist()},
            'SDS':{'lossChanges':[.03,.18,.07,.25],'threshold':.1,'selectedIndices':[2,4]},
            'ASAM':{'fixedRadiusOriginal':.01,'fixedRadiusRescaled':.04,
                    'adaptiveOriginal':.04,'adaptiveRescaled':.04},
            'toyT':toy,'toyTRatios':[1,2,4],
            'fictionalMetrics':{'trainLoss':.0008,'testLoss':.41,'trainAccuracy':1.,'testError':.08,'testAccuracy':1-.08}}

def build():
    vals=numeric_values(); records={}
    for kp in range(13,17):
        stem=f'lec03-bp-v2-kp{kp:02d}-'+{13:'perturbation',14:'rescaling',15:'toy-complexity',16:'metrics'}[kp]
        use_jbd_style(24); fig,ax=new_figure(660,440,margins_px=(100,82,30,35))
        if kp==13:
            angle=np.linspace(0,2*np.pi,600)
            ax.plot(.1*np.cos(angle),.1*np.sin(angle),color=PALETTE['boundary'])
            ax.quiver(0,0,.06,.08,angles='xy',scale_units='xy',scale=1,color=PALETTE['vector'])
            ax.scatter([0,.06],[0,.08],s=85,c=[PALETTE['negative'],PALETTE['positive']])
            ax.annotate('(0.06, 0.08)',(.06,.08),xytext=(-105,16),textcoords='offset points',fontsize=16)
            ax.text(-.115,-.115,r'$\rho=0.1$',fontsize=19)
            ax.set(xlim=(-.13,.13),ylim=(-.13,.13),xlabel=r'$\epsilon_1$',ylabel=r'$\epsilon_2$')
            ax.set_aspect('equal');ax.grid(alpha=.2)
            alt='SAM 的半徑0.1擾動圓，梯度(3,4)方向上的擾動為(0.06,0.08)，長度恰為0.1。'
        elif kp==14:
            x=np.linspace(-.25,.25,600)
            ax.plot(x,x*x,label=r'$L(2+\epsilon)=\epsilon^2$',color=PALETTE['positive'])
            ax.plot(x,4*x*x,label=r'$\widetilde L(1+\epsilon)=4\epsilon^2$',color=PALETTE['negative'])
            ax.scatter([.1,.1,.2],[.01,.04,.04],s=80,c=[PALETTE['positive'],PALETTE['negative'],PALETTE['boundary']])
            ax.set(xlim=(-.25,.25),ylim=(-.005,.26),xlabel=r'$\epsilon$',ylabel='Loss increase')
            ax.legend(fontsize=14,loc='upper center');ax.grid(alpha=.2)
            alt='同一損失經w=2乘重參數化：固定半徑0.1的銳度為0.01與0.04；自適應半徑0.2與0.1都得0.04。'
        elif kp==15:
            y=np.array(vals['toyT'])/1e6
            ax.bar(['A','B','C'],y,color=[PALETTE['positive'],PALETTE['negative'],PALETTE['boundary']])
            for i,v in enumerate(y):ax.text(i,v+.4,f'{v:.2f}',ha='center',fontsize=19)
            ax.set(xlim=(-.7,2.7),ylim=(0,16),xlabel='Toy settings',ylabel=r'$30000\ln(1/\epsilon)/\delta^2$ (millions)')
            ax.grid(axis='y',alpha=.2)
            alt='原祕笈量級玩具式的A、B、C值約3.45、6.91、13.82百萬，比率1:2:4；不是定理給出的實際步數。'
        else:
            ax.bar(['Train','Test'],[.0008,.41],color=[PALETTE['positive'],PALETTE['negative']])
            ax.set_yscale('log');ax.set(xlim=(-.7,1.7),ylim=(.0001,1),xlabel='Fictional source example',ylabel='Loss (log scale)')
            ax.text(0,.0012,'0.0008',ha='center',fontsize=18)
            ax.text(1,.55,'0.41',ha='center',fontsize=18);ax.grid(axis='y',alpha=.2)
            alt='原祕笈假設的訓練損失0.0008與測試損失0.41，對數縱軸顯示兩種評估不同；不是實驗資料。'
        src=svg_data_uri(fig);(OUT/(stem+'.svg')).write_bytes(__import__('base64').b64decode(src.split(',')[1]))
        # Keep the same artists and data, change only colours for the handout.
        fig.patch.set_facecolor('white');ax.set_facecolor('white')
        ax.tick_params(colors='#222222');ax.xaxis.label.set_color('#222222');ax.yaxis.label.set_color('#222222')
        for t in ax.texts:t.set_color('#222222')
        for spine in ax.spines.values():spine.set_color('#444444')
        legend=ax.get_legend()
        if legend:
            for t in legend.get_texts():t.set_color('#222222')
        fig.savefig(OUT/(stem+'.png'),dpi=200,facecolor='white',transparent=False)
        records[kp]={'id':stem,'svg':str(OUT/(stem+'.svg')),'png':stem+'.png','alt':alt,'uri':src,'width':660,'height':440}
        plt.close(fig)
    return records,vals

if __name__=='__main__':
    figures,values=build()
    print(json.dumps(values,ensure_ascii=False,indent=2))
