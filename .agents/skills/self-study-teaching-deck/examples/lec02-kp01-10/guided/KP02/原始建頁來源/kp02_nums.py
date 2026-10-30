from pathlib import Path
import json
ROOT=Path(__file__).parent
DATA=[((0,0),0),((0,1),1),((1,0),1),((1,1),1)]
def train(eta=1,order=DATA):
    w=[0,0,0];trace=[];ends=[]
    for epoch in range(1,21):
        count=0
        for pair,y in order:
            x=[*pair,1];old=w[:];s=sum(a*b for a,b in zip(old,x));o=int(s>=0);e=y-o;dw=[eta*e*a for a in x]
            w=[a+b for a,b in zip(old,dw)]
            sp=sum(a*b for a,b in zip(w,x));norm=sum(a*a for a in x)
            assert sp==s+eta*e*norm
            trace.append(dict(epoch=epoch,pair=list(pair),x=x,y=y,old=old,s=s,o=o,e=e,dw=dw,new=w[:],s_after=sp,norm2=norm))
            count+=int(e!=0)
        ends.append(dict(epoch=epoch,w=w[:],updates=count))
        if count==0:break
    return dict(eta=eta,trace=trace,updates=[r for r in trace if r['e']],passes=ends,final=w)
ONE=train();TWO=train(2)
assert len(ONE['updates'])==5 and len(ONE['passes'])==4 and ONE['final']==[1,1,-1]
assert [r['new'] for r in ONE['updates']]==[[0,0,-1],[0,1,0],[0,1,-1],[1,1,0],[1,1,-1]]
assert [r['updates'] for r in ONE['passes']]==[2,2,1,0]
assert [r['o'] for r in ONE['trace']]==[r['o'] for r in TWO['trace']]
assert all(b['new']==[2*x for x in a['new']] for a,b in zip(ONE['trace'],TWO['trace']))
def save():
    (ROOT/'數值驗算.json').write_text(json.dumps(dict(data=DATA,eta1=ONE,eta2=TWO,convention='s >= 0 => o=1; weights ordered (w1,w2,w0); x=(x1,x2,1)'),ensure_ascii=False,indent=2))
    print('PASS: all 16 sample visits, five updates, pass counts [2,2,1,0], final [1,1,-1], same predictions for eta=1 and eta=2.')
if __name__=='__main__':save()
