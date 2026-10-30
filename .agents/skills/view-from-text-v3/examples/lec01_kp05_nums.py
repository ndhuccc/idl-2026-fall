"""KP05 數字(講義 §1 的 8 點補充例、§3 的兩步回報)。"""
C1 = [(-3, -1.2), (3, -1.2), (-3, -0.8), (3, -0.8)]
C2 = [(-3, 0.8), (3, 0.8), (-3, 1.2), (3, 1.2)]
P8 = C1 + C2
n = len(P8)
mean = tuple(sum(p[k] for p in P8) / n for k in (0, 1)); assert mean == (0, 0)
def cov(ps, m): return [[sum((p[i] - m[i]) * (p[j] - m[j]) for p in ps) / n for j in (0, 1)] for i in (0, 1)]
C = cov(P8, mean)
assert [round(C[0][0], 2), round(C[1][1], 2), round(C[0][1], 2)] == [9.0, 1.04, 0.0]                  # C = diag(9, 1.04)
m1 = tuple(sum(p[k] for p in C1) / 4 for k in (0, 1)); m2 = tuple(sum(p[k] for p in C2) / 4 for k in (0, 1))
assert (m1, m2) == ((0, -1.0), (0, 1.0))
SW = [[(sum((p[i] - m1[i]) * (p[j] - m1[j]) for p in C1) + sum((p[i] - m2[i]) * (p[j] - m2[j]) for p in C2)) / n for j in (0, 1)] for i in (0, 1)]
assert [round(SW[0][0], 2), round(SW[1][1], 2)] == [9.0, 0.04]                                         # S_W = diag(9, 0.04)
SB = [[(4 * (m1[i] - mean[i]) * (m1[j] - mean[j]) + 4 * (m2[i] - mean[i]) * (m2[j] - mean[j])) / n for j in (0, 1)] for i in (0, 1)]
assert [round(SB[0][0], 2), round(SB[1][1], 2)] == [0.0, 1.0]                                          # S_B = diag(0, 1)
J = lambda w: (w[0] ** 2 * SB[0][0] + w[1] ** 2 * SB[1][1]) / (w[0] ** 2 * SW[0][0] + w[1] ** 2 * SW[1][1])
assert round(J((0, 1)), 6) == 25.0 and J((1, 0)) == 0.0
varw = lambda w: w[0] ** 2 * C[0][0] + w[1] ** 2 * C[1][1]
assert varw((1, 0)) == 9.0 and round(varw((0, 1)), 2) == 1.04 and varw((1, 0)) > varw((0, 1))        # PCA 選 (1, 0)
r1, r2 = {"A": 1, "B": 2}, {"A": 3, "B": -5}
G = {k: r1[k] + r2[k] for k in r1}
assert G == {"A": 4, "B": -3} and r1["B"] > r1["A"] and G["A"] > G["B"]                              # 即時 B 較高,整段 A 較高
