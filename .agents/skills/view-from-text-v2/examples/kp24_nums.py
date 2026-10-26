import os
import numpy as np
from math import comb
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kp23_nums import moons2
def Pc(L, p):
    lc = L // 2 + 1 if L % 2 == 0 else (L + 1) // 2
    return sum(comb(L, m) * p ** m * (1 - p) ** (L - m) for m in range(lc, L + 1))
def prod_sum(P):  # P: (L, M)
    P = np.asarray(P, float); pr = np.prod(P, 0); return pr, P.mean(0)
def vdiv(preds):  # preds (L, N) hard labels
    preds = np.asarray(preds); maj = (preds.mean(0) > 0.5).astype(int); return float((preds != maj).mean())
if __name__ == "__main__":
    print("Pc(3,.7)", Pc(3, .7), "Pc(5,.7)", Pc(5, .7))
    for p in (0.6, 0.55, 0.45, 0.5):
        print(p, [round(Pc(L, p), 4) for L in (1, 3, 5, 11, 21, 51)])
    # veto
    P = [[0.7, 0.3]] * 4 + [[0.001, 0.999]]; pr, sm = prod_sum(P); print("prod", pr, "sum", sm, "median", np.median(np.array(P), 0))
    # complementary info
    Xtr, ytr = moons2(100, 1, 0.35); Xte, yte = moons2(1000, 2, 0.35)
    cl = {"logistic": LogisticRegression(), "tree": DecisionTreeClassifier(max_depth=4, random_state=0), "3-NN": KNeighborsClassifier(3)}
    pr_ = {k: m.fit(Xtr, ytr).predict(Xte) for k, m in cl.items()}; acc = {k: (v == yte).mean() for k, v in pr_.items()}; print(acc)
    best = max(acc, key=acc.get); wrong = pr_[best] != yte; other_ok = np.zeros(len(yte), bool)
    for k, v in pr_.items():
        if k != best: other_ok |= (v == yte)
    print("best", best, "wrong", wrong.mean(), "wrong but another right", (wrong & other_ok).mean(), "share of its errors", (wrong & other_ok).sum() / wrong.sum())
    allp = np.array(list(pr_.values())); vote = (allp.mean(0) > 0.5).astype(int); print("vote of 3", (vote == yte).mean(), "V", vdiv(allp))
    # bagged trees independence
    S = []
    for s in range(20):
        X, y = moons2(100, 100 + s, 0.35)
        bg = BaggingClassifier(DecisionTreeClassifier(criterion="entropy"), n_estimators=25, random_state=0).fit(X, y)
        ind = np.array([e.predict(Xte[:, bg.estimators_features_[i]] if False else Xte) for i, e in enumerate(bg.estimators_)]); pind = (ind == yte).mean(1)
        vt = (ind.mean(0) > 0.5).astype(int); S.append((pind.mean(), (vt == yte).mean(), Pc(25, pind.mean()), vdiv(ind)))
    S = np.array(S); print("bagged: mean single", S[:, 0].mean(), "vote", S[:, 1].mean(), "independent theory", S[:, 2].mean(), "V", S[:, 3].mean())
    # diversity: same features bootstrap vs random feature subsets (10-D data)
    rng = np.random.default_rng(0)
    def data(n, seed):
        r = np.random.default_rng(seed); y = r.integers(0, 2, n); X = r.normal(0, 1, (n, 10)) + y[:, None] * 0.9 * np.ones(10)[None, :] * (np.arange(10)[None, :] % 2 == 0) ; return X, y
    res = {"bootstrap": [], "subspace": [], "different types": []}; V = {k: [] for k in res}; single = []
    for s in range(15):
        X, y = data(150, 200 + s); Xt, yt = data(2000, 999)
        preds = []
        for i in range(11):
            idx = np.random.default_rng(s * 100 + i).integers(0, 150, 150); preds.append(DecisionTreeClassifier(max_depth=3, random_state=i).fit(X[idx], y[idx]).predict(Xt))
        P1 = np.array(preds); res["bootstrap"].append(((P1.mean(0) > .5) == yt).mean()); V["bootstrap"].append(vdiv(P1)); single.append((P1 == yt).mean())
        preds = []
        for i in range(11):
            f = np.random.default_rng(s * 100 + i).choice(10, 4, replace=False); preds.append(DecisionTreeClassifier(max_depth=3, random_state=i).fit(X[:, f], y).predict(Xt[:, f]))
        P2 = np.array(preds); res["subspace"].append(((P2.mean(0) > .5) == yt).mean()); V["subspace"].append(vdiv(P2))
        ms = [LogisticRegression(max_iter=500), DecisionTreeClassifier(max_depth=3, random_state=0), KNeighborsClassifier(5), KNeighborsClassifier(15), DecisionTreeClassifier(max_depth=5, random_state=1)]
        P3 = np.array([m.fit(X, y).predict(Xt) for m in ms]); res["different types"].append(((P3.mean(0) > .5) == yt).mean()); V["different types"].append(vdiv(P3))
    print("single tree acc", np.mean(single)); 
    for k in res: print(k, "vote", round(np.mean(res[k]), 4), "+-", round(np.std(res[k]), 4), "V", round(np.mean(V[k]), 4))
