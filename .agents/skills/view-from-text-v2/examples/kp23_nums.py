import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
def H(c):
    c = np.asarray(c, float); n = c.sum(); p = c[c > 0] / n; return float(-(p * np.log2(p)).sum())
def dI(parent, yes, no):
    n = sum(parent); return H(parent) - sum(yes) / n * H(yes) - sum(no) / n * H(no)
def moons2(n, seed, noise=0.35):
    rng = np.random.default_rng(seed); t = rng.uniform(0, np.pi, n)
    a = np.c_[np.cos(t), np.sin(t)]; b = np.c_[1 - np.cos(t), 0.5 - np.sin(t)]
    X = np.r_[a, b] + rng.normal(0, noise, (2 * n, 2)); return X, np.r_[np.zeros(n), np.ones(n)]
if __name__ == "__main__":
    print("I(4,4,2)", H([4, 4, 2]), "I(3,1)", H([3, 1]), "I(1,3,2)", H([1, 3, 2]), "dI", dI([4, 4, 2], [3, 1, 0], [1, 3, 2]))
    print("max entropy 3 classes", H([1, 1, 1]), "pure", H([5, 0, 0]))
    # weighting
    print("A single pure + 9", dI([5, 5], [1, 0], [4, 5]), "naive", H([5, 5]) - (H([1, 0]) + H([4, 5])) / 2)
    print("B (4,1)/(1,4)", dI([5, 5], [4, 1], [1, 4]), "naive", H([5, 5]) - (H([4, 1]) + H([1, 4])) / 2, H([4, 1]), H([4, 5]))
    # practice
    print("practice A (3,1)/(1,3)", dI([4, 4], [3, 1], [1, 3]), "B (2,2)/(2,2)", dI([4, 4], [2, 2], [2, 2]), "C (4,0)/(0,4)", dI([4, 4], [4, 0], [0, 4]))
    # depth sweep
    Xtr, ytr = moons2(100, 1); Xte, yte = moons2(1000, 2)
    for d in (1, 2, 3, 4, 6, 8, 12, None):
        t = DecisionTreeClassifier(criterion="entropy", max_depth=d, random_state=0).fit(Xtr, ytr); print("depth", d, "leaves", t.get_n_leaves(), "train", t.score(Xtr, ytr), "test", t.score(Xte, yte))
    # gini vs entropy
    for crit in ("entropy", "gini"):
        accs = [DecisionTreeClassifier(criterion=crit, max_depth=4, random_state=0).fit(*moons2(100, s)).score(Xte, yte) for s in range(10, 30)]; print(crit, "mean", np.mean(accs), "sd", np.std(accs))
    # pruning ccp
    full = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(Xtr, ytr); path = full.cost_complexity_pruning_path(Xtr, ytr)
    for a in (0.0, 0.002, 0.005, 0.01, 0.02, 0.05):
        t = DecisionTreeClassifier(criterion="entropy", ccp_alpha=a, random_state=0).fit(Xtr, ytr); print("ccp", a, "leaves", t.get_n_leaves(), "train", t.score(Xtr, ytr), "test", t.score(Xte, yte))
    # variance
    roots = []
    for s in range(8):
        X, y = moons2(100, 40 + s); t = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(X, y); roots.append((int(t.tree_.feature[0]), round(float(t.tree_.threshold[0]), 2), int(t.get_n_leaves()), round(t.score(Xte, yte), 3)))
    print("roots", roots)
    single, bag, rf = [], [], []
    for s in range(20):
        X, y = moons2(100, 100 + s)
        single.append(DecisionTreeClassifier(criterion="entropy", random_state=0).fit(X, y).score(Xte, yte))
        bag.append(BaggingClassifier(DecisionTreeClassifier(criterion="entropy"), n_estimators=25, random_state=0).fit(X, y).score(Xte, yte))
        rf.append(RandomForestClassifier(n_estimators=25, criterion="entropy", max_features=1, random_state=0).fit(X, y).score(Xte, yte))
    print("single", np.mean(single), np.std(single), "bag", np.mean(bag), np.std(bag), "rf", np.mean(rf), np.std(rf))
    # oblique
    rng = np.random.default_rng(5); X = rng.uniform(-1, 1, (300, 2)); y = (X[:, 0] + X[:, 1] > 0).astype(int); Xt = rng.uniform(-1, 1, (3000, 2)); yt = (Xt[:, 0] + Xt[:, 1] > 0).astype(int)
    for d in (1, 2, 3, 4, 6, None):
        t = DecisionTreeClassifier(criterion="entropy", max_depth=d, random_state=0).fit(X, y); print("oblique depth", d, "leaves", t.get_n_leaves(), "train", t.score(X, y), "test", t.score(Xt, yt))
    # candidates
    xs = np.array([1.0, 1.4, 2.2, 2.8, 3.5, 4.1, 5.0, 5.5, 6.2, 7.0]); print("candidates", len(np.unique(xs)) - 1, ((np.unique(xs)[1:] + np.unique(xs)[:-1]) / 2)[:5])
