"""KP12 numbers (recomputed)."""
import numpy as np
def image(col):
    x = np.zeros((6, 6), int); x[1:4, col] = 1; return x
K = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
def fmap(x, k=K, b=0):
    n = x.shape[0] - 2; return np.array([[b + int((k * x[i:i + 3, j:j + 3]).sum()) for j in range(n)] for i in range(n)])
def params(side, win=3):
    n = side - win + 1; pos = n * n; return n, pos, pos * (win * win + 1), win * win + 1
if __name__ == "__main__":
    x = image(1); tx = image(3); print(fmap(x)); print(fmap(tx)); print(fmap(x).max(), fmap(tx).max())
    print(params(6), params(8)); print(0.2 * 3 + (-0.1) * 1, 1 - 0.1 * 0.5)
    print(np.array_equal(fmap(tx)[:, 2:], fmap(x)[:, :2]))
