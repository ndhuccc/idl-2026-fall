"""KP05 numbers (recomputed by program)."""
import numpy as np
PTS = [(0, 0), (0, 1), (1, 0), (1, 1)]; TRUTH = {(0, 0): "B", (0, 1): "A", (1, 0): "A", (1, 1): "B"}   # XOR: (0,1),(1,0) are A
def f(t): return 1 if t > 0 else 0
def s1(p): return p[0] + p[1] - 0.5           # master score
def z1(p): return f(s1(p))
def s2(p): return 1.5 - p[0] - p[1]           # ancillary score
def z2(p): return f(s2(p))
def code(p): return f"{z1(p)}{z2(p)}"
def s3(c): return int(c[0]) + int(c[1]) - 1.5   # new master score on the code plane
def pred3(c): return "A" if s3(c) > 0 else "B"
def master_correct():
    # the 1 group is called A, the 0 group B
    return sum(("A" if z1(p) else "B") == TRUTH[p] for p in PTS)
def mixed_count(keyf):
    g = {}
    for p in PTS: g.setdefault(keyf(p), []).append(TRUTH[p])
    return sum(len(v) for v in g.values() if len(set(v)) > 1), g
def final_correct(): return sum(pred3(code(p)) == TRUTH[p] for p in PTS)
def nearest_x1(u=(0, 0), v=(1, 0)):
    a = 2 * (np.array(v) - np.array(u)); b = (np.array(v) @ np.array(v)) - (np.array(u) @ np.array(u)); return a, b, b / a[0]
def pairs(n): return n * (n - 1) // 2
if __name__ == "__main__":
    print([(p, s1(p), z1(p), s2(p), z2(p), code(p), s3(code(p)), pred3(code(p))) for p in PTS])
    print("master correct", master_correct(), "mixed z1", mixed_count(z1), "mixed code", mixed_count(code), "final", final_correct())
    print(nearest_x1(), [pairs(n) for n in (4, 10, 100)], (0.25 + 0.49) ** .5)
