from pathlib import Path
import ast,json,os
os.environ.setdefault("MPLCONFIGDIR","/tmp/idl-kp05-mpl")
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import label
R=Path(__file__).parent
ROWS=[]
for a,b in ((0,0),(0,1),(1,0),(1,1)):
 s1=a+b-1;s2=-a-b+1.5;h1=int(s1>=0);h2=int(s2>=0);s3=h1+h2-2;o=int(s3>=0)
 ROWS.append(dict(x=[a,b],s1=s1,h1=h1,s2=s2,h2=h2,s3=s3,o=o))
assert [v['o'] for v in ROWS]==[0,1,1,0]
A1=np.array([[1,1],[-1,-1]]);b1=np.array([-1,1.5]);A2=np.array([[1,1]]);b2=np.array([-2])
assert np.allclose(A2@A1,0) and np.allclose(A2@b1+b2,-1.5)
for r in ROWS:assert np.allclose(A2@(A1@r['x']+b1)+b2,-1.5)
# General all-affine is not necessarily constant. Validate a nonconstant three-layer example.
rr=np.random.default_rng(11);AA=[rr.normal(size=(3,2)),rr.normal(size=(4,3)),rr.normal(size=(1,4))];BB=[rr.normal(size=(3,)),rr.normal(size=(4,)),rr.normal(size=(1,))];XX=rr.normal(size=(50,2));Z=XX.copy()
for a,b in zip(AA,BB):Z=Z@a.T+b
V=AA[2]@AA[1]@AA[0];c=AA[2]@AA[1]@BB[0]+AA[2]@BB[1]+BB[2]
assert np.max(np.abs(Z-(XX@V.T+c)))<1e-12 and np.ptp(Z)>.1
# Isolate original net_out and reproduce the exact source seed, dimensions and 300x300 grid.
a=ast.parse((R/'source-figures.py').read_text());f=next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name=='net_out');ctx={'np':np};exec(compile(ast.Module(body=[f],type_ignores=[]),'source-figures.py','exec'),ctx);net_out=ctx['net_out']
rng=np.random.default_rng(2);W1=rng.normal(size=(2,4));W2=rng.normal(size=(4,3));W3=rng.normal(size=(3,1));g=np.linspace(-3,3,300);gx,gy=np.meshgrid(g,g);X=np.stack([gx.ravel(),gy.ravel()],axis=1)
ID=net_out(X,W1,W2,W3,False).reshape(gx.shape);RE=net_out(X,W1,W2,W3,True).reshape(gx.shape);err=float(np.max(np.abs(ID.ravel()-(X@(W1@W2@W3)).ravel())));assert err<1e-9
EPS=1e-9
PLOTS=[]
for name,z in [('identity',ID),('relu',RE)]:
 fig,ax=plt.subplots();cs=ax.contourf(gx,gy,z,levels=[-1e9,EPS,1e9]);polys=[]
 for index,segs in enumerate(cs.allsegs):
  for seg in segs:
   if len(seg)>=3:polys.append(dict(side=index,points=seg.tolist()))
 lines=ax.contour(gx,gy,z,levels=[EPS]);segments=[s.tolist() for s in lines.allsegs[0] if len(s)>40]
 # Exactly retain the source's >40-point contour filtering, no hard-coded number of regions.
 assert all(-3-1e-8<=v<=3+1e-8 for p in polys for xy in p['points'] for v in xy)
 counts={str(i):int(label((z>EPS)==bool(i),structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))[1]) for i in (0,1)}
 PLOTS.append(dict(name=name,polygons=polys,segments=segments,components_4_neighbor=counts))
 plt.close(fig)
assert sum(1 for p in PLOTS[1]['polygons'] if p['side']==1)>0
# Reduce only collinear contour vertices so geometry stays exact and under native 200-point limits after splitting.
def simplify(points,tol=1e-10):
 p=np.array(points);out=[]
 for q in p:
  while len(out)>=2 and abs(((out[-1][0]-out[-2][0])*(q[1]-out[-1][1])-(out[-1][1]-out[-2][1])*(q[0]-out[-1][0])))<tol:out.pop()
  out.append(q.tolist())
 return out
for p in PLOTS:
 for poly in p['polygons']:poly['points']=simplify(poly['points'])
 p['segments']=[simplify(s) for s in p['segments']]
def save():
 d={'xor_rows':ROWS,'constant_example':{'A2A1':(A2@A1).tolist(),'bias':(A2@b1+b2).tolist()},'general_affine_max_error':float(np.max(np.abs(Z-(XX@V.T+c)))),'random_experiment':{'seed':2,'dimensions':[2,4,3,1],'no_bias':True,'hidden_activation_only':True,'output_linear':True,'grid_size':300,'range':[-3,3],'display_threshold':EPS,'identity_equivalence_max_error':err,'zero_relu_samples':int(np.sum(RE==0)),'total_samples':90000,'W1':W1.tolist(),'W2':W2.tolist(),'W3':W3.tolist()},'plots':PLOTS,'all_assertions_passed':True}
 (R/'數值驗算.json').write_text(json.dumps(d,ensure_ascii=False,indent=2));print('PASS: four XOR rows; cancelling affine example; general affine composition; source seed2 identity/ReLU experiment.')
 print('Contours:',[(p['name'],p['components_4_neighbor'],[len(s) for s in p['segments']],len(p['polygons'])) for p in PLOTS]);print('Zero ReLU samples:',np.sum(RE==0),'of',RE.size)
if __name__=='__main__':save()
