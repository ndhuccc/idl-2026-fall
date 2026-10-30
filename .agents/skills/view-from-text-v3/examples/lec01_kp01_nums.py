"""KP01 數字:全由程式算出並 assert(對照講義 §3、§5)。"""
from fractions import Fraction as F
lam = F(1, 10)
EA, kA = F(1, 2), 10          # candidate A: training error 0.5, 10 parameters
EB, kB = F(1), 2              # candidate B: training error 1.0, 2 parameters
penA, penB = lam * kA, lam * kB
SA, SB = EA + penA, EB + penB
assert (penA, penB) == (F(1), F(1, 5))
assert (SA, SB) == (F(3, 2), F(6, 5))
assert EA < EB and SB < SA          # training error alone prefers A; the penalty flips it to B
mn = min(SA, SB); arg = "A" if SA < SB else "B"
assert (float(mn), arg) == (1.2, "B")
f = lambda v: f"{float(v):.1f}"
