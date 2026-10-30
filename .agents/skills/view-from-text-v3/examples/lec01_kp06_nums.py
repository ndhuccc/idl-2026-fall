"""KP06 數字(講義 §2 圓度例與平手例、§3 LDA/QDA 邊界)。"""
import math
pdf = lambda x, mu, s: math.exp(-0.5 * ((x - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))
# --- §2 圓度例:蘋果 mu=0.8、黃瓜 mu=0.2、sigma=0.1、先驗各 0.5、x=0.7
pi1 = pi2 = 0.5; x0 = 0.7
f1, f2 = pdf(x0, 0.8, 0.1), pdf(x0, 0.2, 0.1)
assert abs(1 / (0.1 * math.sqrt(2 * math.pi)) - 3.989422804) < 1e-9
assert abs(f1 - 2.419707245) < 1e-8 and abs(f2 - 0.000014867195) < 1e-12
q1, q2 = pi1 * f1, pi2 * f2; fx = q1 + q2
assert abs(q1 - 1.209853623) < 1e-8 and abs(q2 - 0.000007433598) < 1e-12 and abs(fx - 1.209861056) < 1e-8
post1, post2 = q1 / fx, q2 / fx
assert abs(post1 - 0.999993856) < 1e-9 and abs(post2 - 0.000006144) < 1e-9 and abs(post1 + post2 - 1) < 1e-12
# --- 平手例:mu=3,7、sigma=1、x=5
t1, t2 = pdf(5, 3, 1), pdf(5, 7, 1)
assert abs(t1 - 0.053990967) < 1e-9 and t1 == t2
tq = 0.5 * t1; assert abs(tq - 0.026995483) < 1e-9 and abs(tq / (2 * tq) - 0.5) < 1e-12
# --- §3 受控 2D:LDA 邊界 x1=0;QDA 邊界(對數分數相等)
m1, m2 = (-1.0, 0.0), (1.0, 0.0)
def delta(x, m, S):
    return -0.5 * sum((x[i] - m[i]) ** 2 / S[i] for i in (0, 1)) - 0.5 * math.log(S[0] * S[1]) + math.log(0.5)
LDA_S = (1.0, 0.5)
assert all(abs(delta((0, y / 10), m1, LDA_S) - delta((0, y / 10), m2, LDA_S)) < 1e-12 for y in range(-20, 21))   # x1=0 是等後驗邊界
Q1, Q2 = (0.4, 0.8), (1.8, 0.3)
def qgap(x1, x2): return delta((x1, x2), m1, Q1) - delta((x1, x2), m2, Q2)
def qda_curve(y_range=(-1.8, 1.8), x_range=(-3.0, 3.0)):
    """Zero set of qgap: for every x2 row find sign changes along x1 -> list of branches of (x1, x2)."""
    out = []
    ys = [y_range[0] + i * (y_range[1] - y_range[0]) / 60 for i in range(61)]
    for y in ys:
        xs = [x_range[0] + i * (x_range[1] - x_range[0]) / 600 for i in range(601)]
        for a, b in zip(xs, xs[1:]):
            ga, gb = qgap(a, y), qgap(b, y)
            if ga == 0 or ga * gb < 0: out.append((a + (b - a) * ga / (ga - gb) if ga != gb else a, y))
    return out
QPTS = qda_curve()
assert len(QPTS) > 40
