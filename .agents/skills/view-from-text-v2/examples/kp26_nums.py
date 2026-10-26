import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
def make(nmaj, nmin, d, seed):
    """2-D Gaussians; majority at origin, minority at (d,0); unit covariance. labels 0 maj, 1 min."""
    r = np.random.default_rng(seed); X = np.r_[r.normal(0, 1, (nmaj, 2)), r.normal(0, 1, (nmin, 2)) + [d, 0]]; y = np.r_[np.zeros(nmaj), np.ones(nmin)]; return X, y
def metrics(y, p):
    tp = ((p == 1) & (y == 1)).sum(); fn = ((p == 0) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); tn = ((p == 0) & (y == 0)).sum()
    return dict(acc=(tp + tn) / len(y), rec=tp / max(tp + fn, 1), prec=tp / max(tp + fp, 1), fp=int(fp), tp=int(tp))
def avg(f, n=20):
    rs = [f(s) for s in range(n)]; return {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
def oversample(X, y, seed):
    r = np.random.default_rng(seed); mi = np.where(y == 1)[0]; ma = np.where(y == 0)[0]; add = r.choice(mi, len(ma) - len(mi)); i = np.r_[np.arange(len(y)), add]; return X[i], y[i]
def undersample(X, y, seed):
    r = np.random.default_rng(seed); mi = np.where(y == 1)[0]; ma = np.where(y == 0)[0]; keep = r.choice(ma, len(mi), replace=False); i = np.r_[mi, keep]; return X[i], y[i]
def focused_under2(X, y, seed, keep_ratio=5.0):
    """boundary-focused: fit a first classifier, keep the majority points closest to its boundary (largest decision value)."""
    mi = np.where(y == 1)[0]; ma = np.where(y == 0)[0]; d = SVC(kernel="rbf", gamma=0.5, C=1.0).fit(X, y).decision_function(X[ma]); keep = ma[np.argsort(-d)[:int(len(mi) * keep_ratio)]]; i = np.r_[mi, keep]; return X[i], y[i]
def focused_under(X, y, seed, keep_ratio=1.0):
    """keep the majority points closest to the minority class (near the boundary), as many as minority*keep_ratio."""
    mi = np.where(y == 1)[0]; ma = np.where(y == 0)[0]; c = X[mi].mean(0); dist = np.linalg.norm(X[ma] - c, axis=1); keep = ma[np.argsort(dist)[:int(len(mi) * keep_ratio)]]; i = np.r_[mi, keep]; return X[i], y[i]
if __name__ == "__main__":
    Xt, yt = make(5000, 5000, 2.0, 999)
    def run(nmaj, nmin, d, kind, s, clf="svm"):
        X, y = make(nmaj, nmin, d, s)
        if kind == "over": X, y = oversample(X, y, s)
        if kind == "under": X, y = undersample(X, y, s)
        if kind == "focus": X, y = focused_under(X, y, s)
        kw = {"class_weight": "balanced"} if kind == "cost" else {}
        m = SVC(kernel="rbf", gamma=0.5, C=1.0, **kw) if clf == "svm" else DecisionTreeClassifier(min_samples_leaf=5, random_state=0, **kw)
        return metrics(yt, m.fit(X, y).predict(Xt))
    # test set balanced for recall/precision interplay -> use imbalanced test to mirror? use natural 50:1 test instead
    Xt2, yt2 = make(5000, 100, 2.0, 998)
    def run2(nmaj, nmin, d, kind, s, clf="svm", Xe=None, ye=None):
        X, y = make(nmaj, nmin, d, s)
        if kind == "over": X, y = oversample(X, y, s)
        if kind == "under": X, y = undersample(X, y, s)
        if kind == "focus": X, y = focused_under(X, y, s)
        kw = {"class_weight": "balanced"} if kind == "cost" else {}
        m = SVC(kernel="rbf", gamma=0.5, C=1.0, **kw) if clf == "svm" else DecisionTreeClassifier(min_samples_leaf=5, random_state=0, **kw)
        return metrics(ye, m.fit(X, y).predict(Xe))
    print("50:1 (1000 vs 20), d=2 hard:")
    for kind in ("plain", "over", "under", "focus", "cost"): print(kind, avg(lambda s: run2(1000, 20, 2.0, kind, s, "svm", Xt2, yt2)))
    print("tree:"); 
    for kind in ("plain", "over", "under", "cost"): print(kind, avg(lambda s: run2(1000, 20, 2.0, kind, s, "tree", Xt2, yt2)))
    # grid: difficulty x imbalance x size, plain SVM, test balanced-ish: recall on minority
    for d, nm in ((4.0, "easy"), (1.5, "hard")):
        Xe, ye = make(2000, 2000, d, 997)
        for nmaj, nmin in ((100, 100), (1000, 100), (1000, 10), (10000, 100), (10000, 10)):
            print(nm, nmaj, nmin, "recall", round(avg(lambda s: run2(nmaj, nmin, d, "plain", s, "svm", Xe, ye), 8)["rec"], 3))
    # simplicity: C sweep and tree leaf
    for C in (0.01, 0.1, 1, 10, 100):
        r = avg(lambda s: metrics(yt2, SVC(kernel="rbf", gamma=0.5, C=C).fit(*make(1000, 20, 2.0, s)).predict(Xt2)), 10); print("C", C, round(r["rec"], 3), round(r["acc"], 3))
    for leaf in (1, 5, 20, 50):
        r = avg(lambda s: metrics(yt2, DecisionTreeClassifier(min_samples_leaf=leaf, random_state=0).fit(*make(1000, 20, 2.0, s)).predict(Xt2)), 10); print("leaf", leaf, round(r["rec"], 3), round(r["acc"], 3))
    # adaboost cost
    def ada(s, w):
        X, y = make(1000, 20, 2.0, s); sw = np.where(y == 1, w, 1.0); m = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100, random_state=0).fit(X, y, sample_weight=sw / sw.sum()); return metrics(yt2, m.predict(Xt2))
    for w in (1, 10, 50): print("adaboost minority weight", w, avg(lambda s: ada(s, w), 10))
    # practice
    print("all-negative classifier: acc", 1000 / 1010)
