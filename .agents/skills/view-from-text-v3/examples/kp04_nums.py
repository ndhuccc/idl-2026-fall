"""KP04 numbers (recomputed by program)."""
import itertools
TARGETS = ["000", "110", "111"]
CODES = ["".join(c) for c in itertools.product("01", repeat=3)]
def f(t): return 1 if t > 0 else 0
def mism(y, t): return sum(a != b for a, b in zip(y, t))
def h(t, y):
    n1 = t.count("1"); s = [1 if c == "1" else -1 for c in t]
    return sum(si * int(yi) for si, yi in zip(s, y)) - (n1 - 0.5)
def hform(t): return [1 if c == "1" else -1 for c in t], -(t.count("1") - 0.5)
def bits(y): return [f(h(t, y)) for t in TARGETS]
def q(y): return sum(bits(y)) - 0.5
def cls(y): return "A" if q(y) > 0 else "B"
# input plane: three lines g1 = x, g2 = y, g3 = y - x - 1  (bit = step of g)
def plane_code(x, yv): return f"{f(x)}{f(yv)}{f(yv - x - 1)}"
PA = [(1.3, 2.5), (1.4, 0.45), (-1.05, -1.95)]; PB = [(-0.17, 0.4)]
# refine: two overlapping rectangles on a 3x3 grid (col, row)
R1 = {(c, r) for c in (0, 1) for r in (0, 1)}; R2 = {(c, r) for c in (1, 2) for r in (1, 2)}
CELLS = sorted(R1 | R2, key=lambda p: (p[0], p[1]))
def practice():
    t = "011"; return hform(t), h(t, "011"), h(t, "111"), h(t, "010")
if __name__ == "__main__":
    for y in CODES: print(y, [h(t, y) for t in TARGETS], bits(y), q(y), cls(y), [mism(y, "110")], h("110", y))
    print([plane_code(*p) for p in PA], [plane_code(*p) for p in PB], [cls(plane_code(*p)) for p in PA + PB])
    print(len(CELLS), practice())
