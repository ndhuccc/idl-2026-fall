"""KP02 numbers (recomputed): two-layer perceptron on XOR."""
PTS = [(0, 0), (0, 1), (1, 0), (1, 1)]
def g1(a, b): return a + b - 0.5
def g2(a, b): return a + b - 1.5
def step(v): return 1 if v > 0 else 0
def code(a, b): return (step(g1(a, b)), step(g2(a, b)))
def label(a, b): return "A" if a != b else "B"
def gout(y1, y2, bias=-0.5): return y1 - y2 + bias
def pred(a, b, bias=-0.5): return "A" if gout(*code(a, b), bias=bias) > 0 else "B"
def table():
    return [dict(x=p, g1=g1(*p), g2=g2(*p), code=code(*p), truth=label(*p)) for p in PTS]
def no_step(a, b): return (a + b - 0.5) - (a + b - 1.5) - 0.5
def stats(bias):
    pr = [pred(a, b, bias) for a, b in PTS]; wrong = sum(p != label(*q) for p, q in zip(pr, PTS)); return pr, wrong
if __name__ == "__main__":
    for r in table(): print(r)
    print("positions:", sorted({code(*p) for p in PTS}))
    for c in sorted({code(*p) for p in PTS}): print(c, gout(*c))
    print("pred (-0.5):", stats(-0.5), "pred (+0.5):", stats(0.5))
    print("bias +0.5 scores:", [(c, gout(*c, bias=0.5)) for c in sorted({code(*p) for p in PTS})])
    print("no step:", [no_step(*p) for p in PTS], "wrong", sum(("A" != label(*p)) for p in PTS))
    print("(0,1):", g1(0, 1), g2(0, 1), code(0, 1))
    print("pred vector", [1 if pred(*p) == "A" else 0 for p in PTS])
