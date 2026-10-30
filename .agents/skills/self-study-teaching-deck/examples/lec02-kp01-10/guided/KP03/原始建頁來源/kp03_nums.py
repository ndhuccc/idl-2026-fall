from pathlib import Path
import json,math
ROOT=Path(__file__).parent
POINTS=[(0,0),(0,1),(1,0),(1,1)]
LABELS=[-1,1,1,1]
W=(1.,1.);BIAS=-1.5;NORM=math.sqrt(2)
DIST=[dict(x=list(x),y=y,s=sum(a*b for a,b in zip(W,x))+BIAS,d=y*(sum(a*b for a,b in zip(W,x))+BIAS)/NORM) for x,y in zip(POINTS,LABELS)]
assert [round(x['d'],2) for x in DIST]==[1.06,-.35,-.35,.35]
COMB=[dict(y=y,o=o,error=y-o,wrong=y!=o) for y,o in [(1,1),(1,-1),(-1,1),(-1,-1)]]
assert [r['error'] for r in COMB]==[0,2,-2,0]
assert all(r['error']==2*r['y'] for r in COMB if r['wrong'])
NEW=(1.,2.);NEWBIAS=-.5
assert NEW==tuple(w+2*.5*1*x for w,x in zip(W,(0,1))) and NEWBIAS==BIAS+2*.5
OLD_SCORE=-.5;NEW_SCORE=1.5
assert NEW_SCORE-OLD_SCORE==2*.5*(0**2+1**2+1)==2
LOSS=-sum(r['y']*r['s'] for r in DIST if r['d']<0)
assert LOSS==1
assert (1*.5+1*(-.5))==0 # Tangent and normal are orthogonal.
foot=(.25,1.25)
assert sum(foot)==1.5 and math.isclose(math.dist((0,1),foot),abs(DIST[1]['d']))
def loss_at(w,b):return sum(max(0,-y*(sum(a*c for a,c in zip(w,x))+b)) for x,y in zip(POINTS,LABELS))
eps=1e-6
grad=[(loss_at([1+(eps if k==j else 0) for k in range(2)],BIAS)-LOSS)/eps for j in range(2)]
bgrad=(loss_at(W,BIAS+eps)-LOSS)/eps
assert all(math.isclose(a,-1,abs_tol=1e-8) for a in grad) and math.isclose(bgrad,-2,abs_tol=1e-8)
def train(signed=True):
    w=[0.,0.,0.];rows=[];ends=[]
    for epoch in range(1,21):
        count=0
        for pair,ys in zip(POINTS,LABELS):
            x=[*pair,1];y=ys if signed else (ys+1)//2;s=sum(a*b for a,b in zip(w,x));o=(1 if s>=0 else -1) if signed else int(s>=0)
            old=w[:]
            if o!=y:
                coeff=(2*.5*y) if signed else y-o
                w=[a+coeff*b for a,b in zip(w,x)];count+=1
            rows.append(dict(epoch=epoch,x=x,y=y,o=o,s=s,old=old,new=w[:],wrong=o!=y))
        ends.append(dict(epoch=epoch,w=w[:],updates=count))
        if count==0:break
    return dict(visits=rows,passes=ends,updates=[r for r in rows if r['wrong']],final=w)
SIGNED=train();BINARY=train(False)
assert len(SIGNED['updates'])==5 and SIGNED['final']==[1,1,-1]
assert [r['new'] for r in SIGNED['visits']]==[r['new'] for r in BINARY['visits']]
assert [r['updates'] for r in SIGNED['passes']]==[2,2,1,0]
# Tie counterexample: sign(0)=+1 is wrong for a negative label, with distance and loss zero.
TIE=dict(w=[1,1],bias=0,x=[0,0],y=-1,s=0,o=1,d=0,E=0)
assert TIE['y']!=TIE['o'] and TIE['d']==0 and max(0,-TIE['y']*TIE['s'])==0
def save():
    obj=dict(combinations=COMB,distances=DIST,norm=NORM,loss=LOSS,new_w=NEW,new_bias=NEWBIAS,score_before=OLD_SCORE,score_after=NEW_SCORE,signed_training=SIGNED,binary_training=BINARY,boundary_counterexample=TIE)
    (ROOT/'數值驗算.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2))
    print('PASS: 4 label/output cases; 4 exact distances; update (-0.5 -> 1.5); loss 1; 16 identical weight visits; boundary counterexample.')
if __name__=='__main__':save()
