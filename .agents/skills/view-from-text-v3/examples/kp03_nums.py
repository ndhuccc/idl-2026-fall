"""KP03 numbers (recomputed by program)."""
import itertools
CODES = ["".join(c) for c in itertools.product("01", repeat=3)]
def f(t): return 1 if t > 0 else 0
G = [lambda x, y: x, lambda x, y: y, lambda x, y: -x + y - 1]
def code(x, y): return "".join(str(f(g(x, y))) for g in G)
def scores(x, y): return [g(x, y) for g in G]
# first labelling: q = -y1 - y2 + y3 + 0.5, A if q > 0
A1 = [-1, -1, 1, 0.5]
def q1(c): return A1[0] * int(c[0]) + A1[1] * int(c[1]) + A1[2] * int(c[2]) + A1[3]
def cls1(c): return "A" if q1(c) > 0 else "B"
TARGET2 = {"000", "110", "111"}
def cls2(c): return "A" if c in TARGET2 else "B"
def q_gen(a, c): return a[1] * int(c[0]) + a[2] * int(c[1]) + a[3] * int(c[2]) + a[0]   # a = (a0,a1,a2,a3)
def realized(grid=60, lim=4.0):
    s = set()
    for i in range(grid * 2 + 1):
        for j in range(grid * 2 + 1):
            s.add(code(-lim + i * lim / grid, -lim + j * lim / grid))
    return s
def slice_plane(t):
    n1 = t.count("1"); sg = [1 if ch == "1" else -1 for ch in t]; return sg, -(n1 - 0.5)
def h(t, c):
    sg, b = slice_plane(t); return sum(s * int(ch) for s, ch in zip(sg, c)) + b
P, Q, R = (-0.15, -1.0), (-0.15, 2.0), (-0.15, 0.62)
if __name__ == "__main__":
    print(sorted(realized()), set(CODES) - realized())
    print([(c, q1(c), cls1(c)) for c in CODES])
    print(code(1, 1), scores(1, 1), code(-1, -2), code(-2, 2), [code(*p) for p in (P, Q, R)])
    print({t: {c: h(t, c) for c in CODES if h(t, c) > 0} for t in TARGET2})
    print("2nd labelling regions:", {c: cls2(c) for c in CODES})
