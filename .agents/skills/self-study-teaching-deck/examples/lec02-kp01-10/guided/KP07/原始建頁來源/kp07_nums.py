import math,json
from pathlib import Path
hub=lambda e,d=1:.5*e*e if abs(e)<=d else d*(abs(e)-.5*d)
RES=[1,1,1,10];VALS=[[e*e for e in RES],[abs(e) for e in RES],[hub(e) for e in RES]];TOTAL=[sum(v) for v in VALS];SHARE=[v[-1]/sum(v) for v in VALS]
S=sum(math.exp(a) for a in [2,1,0]);PRED=[math.exp(a)/S for a in [2,1,0]];CE=[-math.log(p) for p in PRED];Y=[.7,.2,.1];TERMS=[y*math.log(y/p) for y,p in zip(Y,PRED)];KL=sum(TERMS);H=-sum(y*math.log(y) for y in Y);SOFTCE=-sum(y*math.log(p) for y,p in zip(Y,PRED))
assert TOTAL==[103,13,11.0]
assert abs(CE[1]-1.4076059644443804)<1e-13
assert abs(KL-.005787411901042918)<1e-14
assert abs(SOFTCE-H-KL)<1e-14
assert any(v<0 for v in TERMS) and KL>0
for d in [.5,1,2]:
 for e in [-d,d]:
  assert abs(.5*e*e-d*(abs(e)-.5*d))<1e-14
  assert abs(e-d*(1 if e>0 else -1))<1e-14
assert [max(0,1-s) for s in [1.5,.3,-.5]]==[0,.7,1.5]
FUN={'sq':lambda e:e*e,'abs':abs,'huber':hub,'ce':lambda p:-math.log(p),'hinge':lambda s:max(0,1-s)}
def save():
 data=dict(residuals=RES,values=VALS,sums=TOTAL,means=[a/4 for a in TOTAL],outlier_shares=SHARE,softmax=PRED,exponential_sum=S,cross_entropy=CE,target=Y,kl_terms=TERMS,kl=KL,entropy=H,soft_cross_entropy=SOFTCE,rounded_input_cross_entropy=[-math.log(p) for p in [.665,.245,.090]],rounded_input_kl=sum(y*math.log(y/p) for y,p in zip(Y,[.665,.245,.09])),huber_joins='values and first derivatives match at both thresholds; tested three positive thresholds',source_regression_domain=[-3,3],source_regression_ymax=6,deck_regression_ymax=9.5,source_hinge_domain=[-2,3],source_samples=600)
 (Path(__file__).parent/'數值驗算.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
if __name__=='__main__':save();print('PASS: regression sums/shares, full-precision CE and KL, entropy identity, Huber joins, Hinge examples.')
