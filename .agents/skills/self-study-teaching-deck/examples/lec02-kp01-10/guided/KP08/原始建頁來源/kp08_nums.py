import numpy as np,math,json
from pathlib import Path
OR_X=np.array([[0,0],[0,1],[1,0],[1,1]],float);OR_Y=np.array([0,1,1,1.]);OR_A=np.c_[OR_X,np.ones(4)];ETA=.1
w=np.zeros(3);OR_TRACE=[]
for k in range(13):
 r=OR_Y-OR_A@w;OR_TRACE.append(dict(k=k,w=w.tolist(),prediction=(OR_A@w).tolist(),residual=r.tolist(),loss=float(.5*r@r),gradient=(-OR_A.T@r).tolist()))
 w+=ETA*OR_A.T@r
WSTAR=np.linalg.lstsq(OR_A,OR_Y,rcond=None)[0];YSTAR=OR_A@WSTAR;EMIN=float(.5*np.sum((OR_Y-YSTAR)**2));SPECTRUM=np.linalg.eigvalsh(OR_A.T@OR_A)
assert np.allclose(OR_TRACE[1]['w'],[.2,.2,.3]) and abs(OR_TRACE[1]['loss']-.34)<1e-14
assert np.allclose(WSTAR,[.5,.5,.25]) and abs(EMIN-.125)<1e-14
assert all(OR_TRACE[k+1]['loss']<OR_TRACE[k]['loss'] for k in range(12))
assert 0<ETA<2/max(SPECTRUM)
LX=np.arange(4,dtype=float);LY=np.array([0,1,0,1.]);LXMAT=np.vstack([np.ones(4),LX])
def sigmoid(s):return 1/(1+np.exp(-s))
def ll(beta):
 s=LXMAT.T@np.array(beta);return float(np.sum(LY*s-np.logaddexp(0,s)))
beta=np.zeros(2);NEWTON=[]
for k in range(7):
 pp=sigmoid(LXMAT.T@beta);g=LXMAT@(LY-pp);A=(LXMAT*(pp*(1-pp)))@LXMAT.T
 NEWTON.append(dict(k=k,beta=beta.tolist(),probability=pp.tolist(),loglikelihood=ll(beta),gradient=g.tolist(),curvature=A.tolist()))
 beta+=np.linalg.solve(A,g)
BETASTAR=np.array(NEWTON[-1]['beta']);PSTAR=np.array(NEWTON[-1]['probability']);A0=np.array(NEWTON[0]['curvature']);G0=np.array(NEWTON[0]['gradient'])
assert np.allclose(G0,[0,1]) and np.allclose(A0,[[1,1.5],[1.5,3.5]])
assert abs(np.linalg.det(A0)-1.25)<1e-14 and np.allclose(NEWTON[1]['beta'],[-1.2,.8])
assert all(NEWTON[k+1]['loglikelihood']>=NEWTON[k]['loglikelihood']-1e-14 for k in range(6))
assert np.max(np.abs(LXMAT@(LY-PSTAR)))<1e-12
assert abs(-BETASTAR[0]/BETASTAR[1]-1.5)<1e-12
assert np.count_nonzero((PSTAR>=.5)!=LY)==2
assert np.count_nonzero((LX>=.5)!=LY)==1
WRONG_LL=ll([1.2,-.8]);assert WRONG_LL<ll([0,0])
# Independent finite-difference checks of both objectives and logistic Hessian.
be=np.array([-.7,.4]);eps=1e-5;pg=LXMAT@(LY-sigmoid(LXMAT.T@be));numg=[]
for j in range(2):
 off=np.eye(2)[j]*eps;numg.append((ll(be+off)-ll(be-off))/(2*eps))
assert np.allclose(numg,pg,atol=1e-9)
Wcheck=sigmoid(LXMAT.T@be)*(1-sigmoid(LXMAT.T@be));Hcheck=-(LXMAT*Wcheck)@LXMAT.T
for j in range(2):
 off=np.eye(2)[j]*eps;fd=(LXMAT@(LY-sigmoid(LXMAT.T@(be+off)))-LXMAT@(LY-sigmoid(LXMAT.T@(be-off))))/(2*eps);assert np.allclose(fd,Hcheck[:,j],atol=1e-9)
FUN={'sigmoid':lambda s:float(sigmoid(s)),'sq':lambda e:.5*e*e,'concave':lambda x:1-x*x,'convex':lambda x:x*x}
def save():
 (Path(__file__).parent/'數值驗算.json').write_text(json.dumps(dict(or_trace=OR_TRACE,least_squares_weights=WSTAR.tolist(),least_squares_output=YSTAR.tolist(),least_squares_loss=EMIN,or_hessian_eigenvalues=SPECTRUM.tolist(),eta_upper_bound=2/max(SPECTRUM),newton_trace=NEWTON,wrong_sign_loglikelihood=WRONG_LL,logistic_boundary=-BETASTAR[0]/BETASTAR[1],likelihood_solution_errors=2,alternative_threshold_errors=1,source_curve_grid=dict(domain=[-.5,3.5],samples=300),finite_difference_gradient_and_hessian='passed'),ensure_ascii=False,indent=2))
if __name__=='__main__':save();print('PASS: source OR 12 updates, closed-form minimum, Newton first/two/converged steps, gradient/Hessian finite differences, threshold counterexample.')
