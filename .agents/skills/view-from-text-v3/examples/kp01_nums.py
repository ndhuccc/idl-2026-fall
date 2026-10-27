"""KP01 numbers (recomputed): OR/AND/XOR with one line."""
import itertools
PTS = [(0, 0), (0, 1), (1, 0), (1, 1)]
OR = (0, 1, 1, 1); AND = (0, 0, 0, 1); XOR = (0, 1, 1, 0); XNOR = (1, 0, 0, 1)
def score(w1, w2, w0, p): return w1 * p[0] + w2 * p[1] + w0
def pred(w, p, strict=True):
    v = score(*w, p); return int(v > 0) if strict else int(v >= 0)
def preds(w, strict=True): return tuple(pred(w, p, strict) for p in PTS)
def errors(pr, truth): return sum(a != b for a, b in zip(pr, truth))
LINES = {"or": (1, 1, -0.5), "attempt1": (1, 0, -0.5), "attempt2": (1, 1, -1.5), "attempt3": (1, -1, 0), "and": (1, 1, -1.5)}
def best_over_grid():
    """Smallest number of XOR errors over a grid of (w1, w2, w0) (strict rule): shows 1, never 0."""
    g = [x / 2 for x in range(-8, 9)]; best = 9
    for w in itertools.product(g, repeat=3): best = min(best, errors(preds(w), XOR))
    return best
def separable(truth):
    g = [x / 2 for x in range(-8, 9)]
    return any(errors(preds(w), truth) == 0 for w in itertools.product(g, repeat=3))
if __name__ == "__main__":
    for k, w in LINES.items(): print(k, w, [score(*w, p) for p in PTS], preds(w), "vs OR", errors(preds(w), OR), "vs XOR", errors(preds(w), XOR), "vs AND", errors(preds(w), AND))
    print("accuracy OR line on XOR", 1 - errors(preds(LINES["or"]), XOR) / 4)
    print("best XOR errors over grid", best_over_grid())
    print("separable OR/AND/XOR/XNOR", separable(OR), separable(AND), separable(XOR), separable(XNOR))
    w = LINES["attempt3"]; print("ties x1-x2 strict", preds(w), errors(preds(w), XOR), "ge", preds(w, False), errors(preds(w, False), XOR))
    print("midpoint g(M) for or line", score(*LINES["or"], (0.5, 0.5)))
    print("s values", [p[0] + p[1] for p in PTS])
