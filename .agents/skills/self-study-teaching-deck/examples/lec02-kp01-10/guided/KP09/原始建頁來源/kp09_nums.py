from pathlib import Path
import numpy as np,json,hashlib
ROOT=Path(__file__).parent
X=np.array([[0,0],[0,1],[1,0],[1,1]],float);Y=np.array([0,1,1,0]);MU=np.array([[0,0],[1,1]],float)
D=((X[:,None,:]-MU[None])**2).sum(-1);PHI=np.exp(-D);A=np.column_stack([PHI,np.ones(4)])
B=np.linalg.lstsq(A,Y,rcond=None)[0];BOUND=(.5-B[2])/B[0]
W0=np.array([[.2,.2],[.5,.5],[.8,.8]]);SX=np.array([.9,.7]);DIST=np.linalg.norm(SX-W0,axis=1);W1=W0.copy();W1[1:]+=.5*(SX-W1[1:])
rng=np.random.default_rng(0);N=8;gi,gj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij');GRID0=.5+.06*rng.normal(size=(N,N,2));g=GRID0.copy();SNAPS={0:g.copy()};FIRST=None
for tau in range(4000):
 x=rng.random(2);zi,zj=np.unravel_index(((g-x)**2).sum(-1).argmin(),(N,N));eta=.5*(1-tau/4000)+.01;sig=3*(1-tau/4000)+.4;h=np.exp(-((gi-zi)**2+(gj-zj)**2)/(2*sig**2));g+=eta*h[...,None]*(x-g)
 if tau==0:FIRST=dict(x=x.tolist(),winner=[int(zi),int(zj)],eta=eta,sigma=sig,h=h.tolist())
 if tau+1 in [1,100,1000,4000]:SNAPS[tau+1]=g.copy()
GRID=g.copy()
assert np.allclose(A@B,Y,atol=1e-12) and np.linalg.matrix_rank(A)==3
assert np.allclose(W1,[[.2,.2],[.7,.6],[.85,.75]])
def save():
 d=dict(xor=dict(x=X.tolist(),targets=Y.tolist(),centres=MU.tolist(),two_sigma_squared=1,squared_distances=D.tolist(),features=PHI.tolist(),augmented_design=A.tolist(),least_squares=B.tolist(),prediction=(A@B).tolist(),boundary=BOUND,gram=(A.T@A).tolist(),rank=int(np.linalg.matrix_rank(A))),som_hand=dict(weights_before=W0.tolist(),input=SX.tolist(),euclidean=DIST.tolist(),l1=np.abs(SX-W0).sum(1).tolist(),winner=3,eta=.5,neighbourhood=[2,3],weights_after=W1.tolist()),som_grid=dict(seed=0,generator='numpy.random.default_rng / PCG64',initialization='.5+.06*rng.normal(size=(8,8,2)); consumes random numbers before input draws',updates=4000,eta='.5*(1-tau/4000)+.01',sigma='3*(1-tau/4000)+.4',neighbourhood='Gaussian on fixed grid indices; all nodes, positive weights',first_update=FIRST,snapshots={str(k):v.tolist() for k,v in SNAPS.items()},final_sha256=hashlib.sha256(GRID.tobytes()).hexdigest(),numpy_version=np.__version__))
 (ROOT/'數值驗算.json').write_text(json.dumps(d,ensure_ascii=False,indent=2));return d
if __name__=='__main__':save();print(B,BOUND,DIST)
