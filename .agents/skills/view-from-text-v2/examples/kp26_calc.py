import json, numpy as np
from kp26_nums import *
Xt, yt = make(5000, 100, 2.0, 998)
def counts(p, y):
    return dict(tp=int(((p==1)&(y==1)).sum()), fn=int(((p==0)&(y==1)).sum()), fp=int(((p==1)&(y==0)).sum()), tn=int(((p==0)&(y==0)).sum()))
def mean_counts(f, n=20):
    rs=[f(s) for s in range(n)]; return {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
def svm(C=1.0, **kw): return SVC(kernel="rbf", gamma=0.5, C=C, **kw)
def tree(leaf=5, **kw): return DecisionTreeClassifier(min_samples_leaf=leaf, random_state=0, **kw)
def prep(kind, s, ratio=1.0):
    X, y = make(1000, 20, 2.0, s)
    if kind=="over": X,y=oversample(X,y,s)
    if kind=="under": X,y=undersample(X,y,s)
    if kind=="focus": X,y=focused_under2(X,y,s,ratio)
    return X,y
def run(model, kind="plain", ratio=1.0, Xe=Xt, ye=yt, maj=1000, mn=20, d=2.0):
    def f(s):
        if maj==1000 and mn==20 and d==2.0: X,y=prep(kind,s,ratio)
        else: X,y=make(maj,mn,d,s)
        m=model(); return counts(m.fit(X,y).predict(Xe), ye)
    return mean_counts(f)
R={}
R["plain_svm"]=run(svm)
R["C"]={str(C): run(lambda C=C: svm(C)) for C in (0.01,0.1,1,10,100)}
R["leaf"]={str(L): run(lambda L=L: tree(L)) for L in (1,5,20,50)}
R["grid"]={}
for d,nm in ((4.0,"easy"),(1.5,"hard")):
    Xe,ye=make(2000,2000,d,997)
    for maj,mn in ((100,100),(1000,100),(1000,10),(10000,100),(10000,10)):
        R["grid"][f"{nm}-{maj}:{mn}"]=run(svm,Xe=Xe,ye=ye,maj=maj,mn=mn,d=d)
for k in ("over","under"): R[k+"_svm"]=run(svm,k)
R["cost_svm"]=run(lambda: svm(1.0,class_weight="balanced"))
R["focus"]={str(r): run(svm,"focus",float(r)) for r in (1,3,5,10)}
for k in ("plain","over","under"): R[k+"_tree"]=run(tree,k)
R["cost_tree"]=run(lambda: tree(5,class_weight="balanced"))
def ada(w):
    def f(s):
        X,y=make(1000,20,2.0,s); sw=np.where(y==1,w,1.0)
        m=AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=100,random_state=0).fit(X,y,sample_weight=sw/sw.sum()); return counts(m.predict(Xt),yt)
    return mean_counts(f)
R["ada"]={str(w): ada(w) for w in (1,10,50)}
json.dump(R,open("kp26_cache.json","w"),indent=1)
for k,v in R.items():
    print(k, v if "tp" in v else {a:{x:round(y,2) for x,y in b.items()} for a,b in v.items()})
