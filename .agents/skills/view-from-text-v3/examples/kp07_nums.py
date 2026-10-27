import numpy as np, math
X=np.array([[0,0],[0,1],[1,0],[1,1]],float); Y=np.array([0,1,1,0],float)
W0=np.array([[-2,5,5],[-7,5,5],[-2,8,-12]],float)
sig=lambda v:1/(1+np.exp(-np.clip(v,-40,40)))
def fwd(W,x):
    h=sig(W[:2,0]+W[:2,1:]@x); o=sig(W[2,0]+W[2,1:]@h); return h,o
def evalJ(W):
    J=0;c=0;outs=[]
    for x,y in zip(X,Y):
        h,o=fwd(W,x); J+=0.5*(y-o)**2; c+=(o>=0.5)==y; outs.append(o)
    return J/4,int(c),outs
def grads(W,x,y):
    h,o=fwd(W,x); do=(y-o)*o*(1-o); d=h*(1-h)*do*W[2,1:]
    G=np.zeros((3,3)); G[2]=do*np.array([1,h[0],h[1]]); G[0]=d[0]*np.array([1,*x]); G[1]=d[1]*np.array([1,*x]); return G,do,d,h,o
def batch_epoch(W,eta):
    G=sum(grads(W,x,y)[0] for x,y in zip(X,Y)); return W+eta*G/4,G
def train_batch(W,eta,maxE=60000,tol=2e-4):
    W=W.copy(); hist=[evalJ(W)[0]]
    for e in range(1,maxE+1):
        W,_=batch_epoch(W,eta); J,c,_=evalJ(W); hist.append(J)
        if c==4 and J<tol: return W,hist,e
    return W,hist,None
def train_pattern(W,eta,order=None,maxE=60000,tol=2e-4):
    W=W.copy(); hist=[evalJ(W)[0]]; idx=[0,1,2,3] if order is None else order
    for e in range(1,maxE+1):
        for i in idx:
            G=grads(W,X[i],Y[i])[0]; W=W+eta*G
        J,c,_=evalJ(W); hist.append(J)
        if c==4 and J<tol: return W,hist,e
    return W,hist,None
if __name__=="__main__":
    J0,c0,o0=evalJ(W0); print('init J',J0,c0,[round(o,4) for o in o0])
    for x,y in zip(X,Y):
        G,do,d,h,o=grads(W0,x,y); print(x,y,'h',h.round(4),'o',round(o,4),'do',round(do,4),'d',d.round(5))
    Wn,G=batch_epoch(W0,1.5); print('G sum',G.round(5)); print('dW',(1.5*G/4).round(5)); print('after1',evalJ(Wn)[:2])
    for eta in (0.1,0.5,1.5,3.0):
        W,h,e=train_batch(W0,eta); print('eta',eta,'epochs',e,'final J',h[-1],'J at 1,10,100,1000:',[round(h[k],5) if k<len(h) else None for k in (1,10,100,1000)])
    W,h,e=train_batch(W0,1.5); print('trained W',W.round(3)); print('outs',[round(o,4) for o in evalJ(W)[2]])
    Wp,hp,ep=train_pattern(W0,1.5); print('pattern eta1.5 epochs',ep)
    Wp,hp,ep=train_pattern(W0,0.5); print('pattern eta0.5 epochs',ep)
    # zero init
    Z=np.zeros((3,3)); W,h,e=train_batch(Z,1.5,maxE=20000); print('zero init:',e,h[0],h[100],h[-1]); print(W.round(4))
    # random seeds small init, batch eta 1.5
    res=[]
    for s in range(20):
        r=np.random.RandomState(s); Wr=r.uniform(-0.5,0.5,(3,3)); W,h,e=train_batch(Wr,1.5,maxE=30000); res.append((s,e,round(h[-1],5)))
    print(res)

# ---- vectorised batch / pattern training over many starts (S,3,3) ----
def vfwd(W, X_):                       # W (S,3,3), X_ (4,2) -> h (S,4,2), o (S,4)
    v = W[:, None, :2, 0] + np.einsum("sjk,nk->snj", W[:, :2, 1:], X_); h = sig(v)
    o = sig(W[:, None, 2, 0] + np.einsum("sj,snj->sn", W[:, 2, 1:], h)); return h, o
def vgrad(W):
    h, o = vfwd(W, X); do = (Y[None] - o) * o * (1 - o); d = h * (1 - h) * do[..., None] * W[:, None, 2, 1:]
    G = np.zeros_like(W); G[:, 2, 0] = do.sum(1); G[:, 2, 1:] = np.einsum("sn,snj->sj", do, h)
    for j in range(2): G[:, j, 0] = d[:, :, j].sum(1); G[:, j, 1:] = np.einsum("sn,nk->sk", d[:, :, j], X)
    return G
def vJ(W):
    h, o = vfwd(W, X); return 0.5 * ((Y[None] - o) ** 2).mean(1), ((o >= 0.5) == (Y[None] == 1)).all(1), o
def vtrain(W0_, eta, maxE=60000, tol=2e-4):
    W = W0_.copy(); S = len(W); done = np.zeros(S, int); alive = np.ones(S, bool)
    for e in range(1, maxE + 1):
        W[alive] += eta * vgrad(W[alive]) / 4; J, ok, _ = vJ(W)
        fin = alive & ok & (J < tol); done[fin] = e; alive &= ~fin
        if not alive.any(): break
    return W, done, vJ(W)[0]
if __name__ == "__main__":
    print("---- extra experiments")
    for eta in (0.1, 0.5, 1.5, 3, 6, 30, 100, 200):
        W, d, J = vtrain(W0[None], eta, 80000); print("eta", eta, "epochs", d[0] or "fail", "J", float(J[0]), "outs", np.round(vJ(W)[2][0], 2))
    Wr = np.stack([np.random.RandomState(s).uniform(-3, 3, (3, 3)) for s in range(300)])
    W, d, J = vtrain(Wr, 3, 20000); print("300 seeds eta3:", int((d > 0).sum()), "converged;", int((d == 0).sum()), "not; seed0", d[0], float(J[0]), "seed4", d[4], float(J[4]))
    print("seed0 outs", np.round(vJ(W[:1])[2][0], 2), "grad norm", float(np.linalg.norm(vgrad(W[:1]))))
    Wp, hp, ep = train_pattern(W0, 1.5); print("pattern 1.5", ep)
    Wp, hp, ep = train_pattern(Wr[0], 1.5, maxE=20000); print("pattern seed0:", ep, hp[-1])
    Wb, hb, eb = train_batch(W0, 6.0); print("batch 6", eb)
    Wf, hf, ef = train_batch(W0, 1.5); print("frozen", np.round(Wf, 3), ef, [round(o, 4) for o in evalJ(Wf)[2]])
    Wz, hz, ez = train_batch(np.zeros((3, 3)), 1.5, maxE=3000); print("zero", hz[0], hz[-1])
    Ws = np.stack([np.random.RandomState(s).uniform(-0.5, 0.5, (3, 3)) for s in range(20)]); W, d, J = vtrain(Ws, 1.5, 40000); print("20 small seeds", d.min(), d.max(), int((d > 0).sum()))
