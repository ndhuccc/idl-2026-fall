from pathlib import Path
import json,itertools,ast
import numpy as np
ROOT=Path(__file__).parent
XOR=[(0,0,0),(0,1,1),(1,0,1),(1,1,0)]
def run_xor():
 w=np.zeros(3);passes=[];visits=[]
 for ep in range(1,11):
  n=0
  for a,b,y in XOR:
   x=np.array([a,b,1]);s=float(w@x);o=int(s>=0);old=w.copy()
   if o!=y:w+=(y-o)*x;n+=1
   visits.append(dict(epoch=ep,x=[a,b],y=y,s=s,o=o,old=old.tolist(),new=w.tolist(),wrong=o!=y))
  passes.append(dict(epoch=ep,updates=n,w=w.tolist()))
 return passes,visits
PASSES,VISITS=run_xor()
assert [p['updates'] for p in PASSES]==[3,3,4,4,4,4,4,4,4,4]
assert all(p['w']==[-1,0,0] for p in PASSES[3:])
CYCLE=[v for v in VISITS if v['epoch']==4]
assert CYCLE[-1]['new']==CYCLE[0]['old'] and all(v['wrong'] for v in CYCLE)
# Execute exactly the source's isolated data/training function, without plotting or top-level side effects.
a=ast.parse((ROOT/'source-figures.py').read_text());f=next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name=='train_boundaries');ctx={'np':np,'step':lambda s:int(s>=0)}
exec(compile(ast.Module(body=[f],type_ignores=[]),'source-figures.py','exec'),ctx)
DATA,BOUNDARIES=ctx['train_boundaries']();NEW=np.array([2.5,2.4])
BOUNDS=[]
for key in (0,2):
 w,b=BOUNDARIES[key];score=float(w@NEW+b);pred=int(score>=0)
 assert all(int(w@x+b>=0)==y for x,y in DATA)
 BOUNDS.append(dict(name='AB'[(0,2).index(key)],w=w.tolist(),bias=b,score=score,pred=pred))
assert BOUNDS[0]['pred']==1 and BOUNDS[1]['pred']==0
assert np.allclose(np.round(BOUNDS[0]['w'],2),[4.03,3.17])
assert np.allclose(np.round(BOUNDS[1]['w'],2),[.94,.95])
assert round(BOUNDS[0]['bias'],2)==-15.22 and round(BOUNDS[1]['bias'],2)==-5.04
GRID=[(w,sum(int(w[0]*a+w[1]*b+w[2]>=0)==y for a,b,y in XOR)) for w in itertools.product(range(-3,4),repeat=3)]
assert len(GRID)==343 and max(n for _,n in GRID)==3
CANDIDATES=[]
for b in (-.5,-1.5):
 bad=[j for j,(a,c,y) in enumerate(XOR) if int(a+c+b>=0)!=y];CANDIDATES.append(dict(w=[1,1],bias=b,wrong_indices=bad,correct=4-len(bad)))
assert [c['correct'] for c in CANDIDATES]==[3,1]
assert [int(a+b-1.5>=0) for a,b in [(0,0),(0,1),(1,0),(1,1)]]==[0,0,0,1]
# Both XOR class-segments share the midpoint, precluding strict separation.
assert np.allclose((np.array([0,0])+[1,1])/2,(np.array([0,1])+[1,0])/2)
def save():
 result={'source_seed_data':3,'source_seeds_training':[100,102],'xor_passes':PASSES,'xor_visits':VISITS,'cycle_pass4':CYCLE,'training_data':[dict(x=x.tolist(),y=y) for x,y in DATA],'boundaries':BOUNDS,'new_point':NEW.tolist(),'grid':{'range':[-3,3],'weights_tested':343,'max_correct':3,'best_example':next(w for w,n in GRID if n==3)},'candidate_lines':CANDIDATES,'all_assertions_passed':True}
 (ROOT/'數值驗算.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 print('PASS: XOR 10 passes, exact 4-update cycle, 343 grid weights, 24 source-seed points, two zero-error boundaries and opposite predictions.')
 print('Boundaries:',BOUNDS,'Cycle:',[v['new'] for v in CYCLE])
if __name__=='__main__':save()
