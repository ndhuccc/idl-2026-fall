import math,json
from pathlib import Path
sig=lambda a:1/(1+math.exp(-a))
FUN={
'linear':lambda a:a,'sigmoid':sig,'tanh':math.tanh,'gaussian':lambda a:math.exp(-a*a),
'relu':lambda a:max(0,a),'leaky':lambda a:max(.01*a,a),'prelu':lambda a:max(.2*a,a),
'elu':lambda a:a if a>=0 else math.expm1(a),'softplus':lambda a:math.log1p(math.exp(a)),
'maxout':lambda a:abs(a)}
DER={'linear':lambda a:1,'sigmoid':lambda a:sig(a)*(1-sig(a)),'tanh':lambda a:1-math.tanh(a)**2,'gaussian':lambda a:-2*a*math.exp(-a*a),'relu':lambda a:1 if a>0 else 0,'leaky':lambda a:1 if a>0 else .01,'prelu':lambda a:1 if a>0 else .2,'elu':lambda a:1 if a>=0 else math.exp(a),'softplus':sig}
EXPS=[math.exp(a) for a in (2,1,0)];SUM=sum(EXPS);PROBS=[e/SUM for e in EXPS]
NUMS={k:{str(a):dict(value=FUN[k](a),derivative=DER[k](a)) for a in (-2,2)} for k in ['relu','leaky','elu','softplus']}
assert abs(SUM-11.107337927389695)<1e-12
assert abs(sum(PROBS)-1)<1e-15
assert abs(PROBS[0]-.6652409557748219)<1e-14
assert abs((.25)**5-.0009765625)<1e-15
assert abs(NUMS['elu']['-2']['value']+.8646647167633873)<1e-14
assert abs(NUMS['softplus']['2']['derivative']-.8807970779778823)<1e-14
# Independent finite-difference verification away from kinks.
for k in DER:
 for a in [-2,-.7,.6,2]:
  fd=(FUN[k](a+1e-5)-FUN[k](a-1e-5))/2e-5
  assert abs(fd-DER[k](a))<2e-8,(k,a,fd)
def save():
 (Path(__file__).parent/'數值驗算.json').write_text(json.dumps(dict(exp=EXPS,total=SUM,probabilities=PROBS,comparison=NUMS,activation_product=.25**5,curve_domain=[-3,3],source_samples=600,derivative_finite_difference='passed for 9 functions at 4 non-kink points each'),ensure_ascii=False,indent=2))
