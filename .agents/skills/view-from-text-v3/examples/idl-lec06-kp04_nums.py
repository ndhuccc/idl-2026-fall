"""KP04 slide numbers (lec06-kp04). Every number on the slides comes from here; the arithmetic reuses the lecture's
figure script assets/figures/src/lec06-kp04-figures.py (same functions, same data), so slides and lecture cannot drift.
    python3 kp04_nums.py          # prints and asserts everything
Exercise answers are asserted here but are NEVER written to a slide or to the speaker notes (hints only)."""
import importlib.util, os, sys
import numpy as np
REPO = os.environ.get("IDL_REPO", "/workspace/idl-2026-fall")
_src = os.path.join(REPO, "assets/figures/src/lec06-kp04-figures.py")
sys.path.insert(0, os.path.dirname(_src))
_spec = importlib.util.spec_from_file_location("lec06kp04fig", _src); F = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(F)
corr1d, conv1d_full, conv1d_valid, corr2d, conv2d_valid, conv2d_full, corr1d_full = F.corr1d, F.conv1d_full, F.conv1d_valid, F.corr2d, F.conv2d_valid, F.conv2d_full, F.corr1d_full

x = np.array([2, 1, 3]); w = np.array([4, 5]); n, k = len(x), len(w); wt = w[::-1]
L = lambda a: [int(v) for v in np.asarray(a).ravel()] if np.ndim(a) == 1 else [[int(v) for v in r] for r in a]

# page 1: weighted sum at the first position, kernel as is vs flipped
p1_as_is = int(x[0] * w[0] + x[1] * w[1]); p1_flip = int(x[0] * w[1] + x[1] * w[0]); valid_n = n - k + 1
# page 2: positions tau of w[tau], w[-tau], w[1-tau]   (dict tau -> weight)
pos_w = {t: int(w[t]) for t in range(k)}; pos_wm = {-t: int(w[t]) for t in range(k)}; pos_w1 = {1 - t: int(w[t]) for t in range(k)}
# page 3: full convolution, term by term
full = L(conv1d_full(x, w)); full_terms = []
for t in range(n + k - 1):
    full_terms.append([(tau, int(x[tau]), t - tau, int(w[t - tau])) for tau in range(n) if 0 <= t - tau < k])
valid_conv = L(conv1d_valid(x, w)); full_n = n + k - 1
# page 4/5: cross-correlation, and with the flipped kernel
corr = L(corr1d(x, w)); corr_flip = L(corr1d(x, wt))
# page 6: 3x3 kernel 1..9, left-right flip, 180-degree rotation
K9 = np.arange(1, 10).reshape(3, 3); K9_lr = L(np.fliplr(K9)); K9_rot = L(np.rot90(K9, 2))
# page 7/8: 2-D main example
I = np.array([[1, 2, 0], [1, 1, 2], [0, 1, 1]]); K = np.eye(2, dtype=int)
s = L(corr2d(I, K)); s_conv = L(conv2d_valid(I, K)); win01 = L(I[0:2, 1:3]); s01_terms = [int(v) for v in (I[0:2, 1:3] * K).ravel()]
Ffull = conv2d_full(I, K); full_block = L(Ffull[1:3, 1:3]); full2_shape = Ffull.shape
# page 10: Prewitt
Hp = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]); Hp_rot = L(np.rot90(Hp, 2))
# page 13: commutativity
full_wx = L(conv1d_full(w, x)); fcorr_xw = L(corr1d_full(x, w)); fcorr_wx = L(corr1d_full(w, x))
# page 14: exercise data (answers asserted, never printed on slides)
e1x, e1w = np.array([1, 0, 2]), np.array([3, 4]); e2X = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]); e3K = np.array([[1, 2], [0, 0]]); e4x = np.array([3, 1, 4, 1, 5])
e1_flip = L(e1w[::-1]); e3_rot = L(np.rot90(e3K, 2))

def check():
    assert (p1_as_is, p1_flip, valid_n) == (13, 14, 2)
    assert pos_w == {0: 4, 1: 5} and pos_wm == {0: 4, -1: 5} and pos_w1 == {1: 4, 0: 5}
    assert full == [8, 14, 17, 15] and valid_conv == [14, 17] and full[1:3] == valid_conv and full_n == 4
    assert full == L(np.convolve(x, w, "full"))
    assert corr == [13, 19] and corr == L(np.correlate(x, w, "valid")) and corr_flip == valid_conv
    assert K9_lr == [[3, 2, 1], [6, 5, 4], [9, 8, 7]] and K9_rot == [[9, 8, 7], [6, 5, 4], [3, 2, 1]] and K9_rot == L(np.flipud(np.fliplr(K9)))
    assert s == [[2, 4], [2, 2]] and s_conv == s and s01_terms == [2, 0, 0, 2] and sum(s01_terms) == 4 and win01 == [[2, 0], [1, 2]]
    assert full_block == s and full2_shape == (4, 4) and L(np.rot90(K, 2)) == L(K)
    assert Hp_rot == L(-Hp)
    assert full_wx == full and fcorr_xw == [10, 13, 19, 12] and fcorr_wx == [12, 19, 13, 10] and fcorr_wx == fcorr_xw[::-1]
    assert e1_flip == [4, 3] and e3_rot == [[0, 0], [2, 1]]
    # answers (kept off the slides): lecture practice 1-4
    assert L(conv1d_valid(e1x, e1w)) == [4, 6] and L(corr1d(e1x, e1w)) == [3, 8]
    assert L(corr2d(e2X, Hp)) == [[-6]] and L(conv2d_valid(e2X, Hp)) == [[6]]
    assert L(corr2d(I, e3K)) == [[5, 2], [3, 5]] and L(conv2d_valid(I, e3K)) == [[3, 4], [1, 3]]
    for kk in (np.array([1, 2, 1]), np.array([1, 0, -1])): assert len(corr1d(e4x, kk)) == 3
    return True

if __name__ == "__main__":
    check()
    for nm in ("p1_as_is", "p1_flip", "valid_n", "pos_w", "pos_wm", "pos_w1", "full", "full_terms", "valid_conv", "corr", "corr_flip", "K9_lr", "K9_rot", "s", "s_conv", "win01", "s01_terms", "full_block", "Hp_rot", "full_wx", "fcorr_xw", "fcorr_wx", "e1_flip", "e3_rot"):
        print(f"{nm:12s}", globals()[nm])
    print("all asserts passed")
