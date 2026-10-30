"""KP07 數字(講義 §1 的獨立二次例、§4 的兩種共變異數分解)。"""
import math
z = (3.0, 0.5); lam = 1.0
ridge = tuple(v / (1 + lam) for v in z); assert ridge == (1.5, 0.25)
soft = lambda v, t: math.copysign(max(abs(v) - t, 0.0), v)
lasso = tuple(soft(v, lam) for v in z); assert lasso == (2.0, 0.0)
# 同一個 d=2,q=1、C=diag(9,1.04) 的兩種分解(KP05 的 8 點)
C = (9.0, 1.04)
fa_W = math.sqrt(8.5); fa_psi = (0.5, 1.04)
assert abs(fa_W - 2.9155) < 1e-4 and abs(fa_W ** 2 + fa_psi[0] - C[0]) < 1e-12 and abs(0 + fa_psi[1] - C[1]) < 1e-12
pp_W = math.sqrt(7.96); pp_psi = (1.04, 1.04)
assert abs(pp_W - 2.8213) < 1e-4 and abs(pp_W ** 2 + pp_psi[0] - C[0]) < 1e-12 and abs(0 + pp_psi[1] - C[1]) < 1e-12
assert fa_psi[0] != fa_psi[1] and pp_psi[0] == pp_psi[1]            # FA 各座標雜訊不同,pPCA 相同
