"""KP04 數字(講義 §2–§4)。"""
import math
P15 = [(1, 2), (1, 3), (2, 2), (4, 2), (4, 3), (5, 2), (5.5, 2), (5.5, 3), (6.5, 2), (10, 2), (10, 3), (11, 2), (13, 2), (13, 3), (14, 2)]
INIT = {2: [(1, 2.5), (13, 2.5)], 3: [(1.33, 2.33), (5.83, 2.33), (13.33, 2.33)], 5: [(1.33, 2.33), (4.33, 2.33), (5.83, 2.33), (10.33, 2.33), (13.33, 2.33)]}
d = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
def kmeans(pts, cen):
    cen = list(cen)
    for _ in range(100):
        asg = [min(range(len(cen)), key=lambda j: d(p, cen[j])) for p in pts]
        new = [tuple(sum(p[k] for p, a in zip(pts, asg) if a == j) / max(1, asg.count(j)) for k in (0, 1)) for j in range(len(cen))]
        if new == cen: break
        cen = new
    return asg, cen
RES = {m: kmeans(P15, c) for m, c in INIT.items()}
sizes = {m: [a.count(j) for j in range(m)] for m, (a, _) in RES.items()}
assert sizes[2] == [9, 6] and sizes[3] == [3, 6, 6] and sizes[5] == [3, 3, 3, 3, 3]          # 講義 §2 的表
A, B = (1, 1), (6, 6)
Q5 = [(1, 1), (2, 2), (1, 3), (6, 6), (7, 5)]
dist = [(round(d(q, A), 4), round(d(q, B), 4)) for q in Q5]
assert dist == [(0.0, 7.0711), (1.4142, 5.6569), (2.0, 5.831), (7.0711, 0.0), (7.2111, 1.4142)]   # 講義 §3 的表
assert round(math.sqrt(50), 4) == 7.0711 and d((6, 6), B) == 0                                    # 來源表的 1.4 不成立
near = ["A" if a <= b else "B" for a, b in dist]; assert near == ["A", "A", "A", "B", "B"]
xs = [p[0] for p in P15]; assert len(set(xs)) == 10 and len(P15) == 15                           # 保留 x1:15 筆落在 10 個位置
from collections import Counter
mult = Counter(xs); assert sorted(mult.values()) == [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
