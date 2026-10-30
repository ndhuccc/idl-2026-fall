"""KP08 數字(講義 §2 XOR 四點、§3 同一已分對樣本)。"""
import itertools
XOR = {(0, 0): 0, (1, 0): 1, (0, 1): 1, (1, 1): 0}
pred = lambda w1, w2, b, x: 1 if w1 * x[0] + w2 * x[1] + b > 0 else 0
grid = [i / 4 for i in range(-24, 25)]
assert not any(all(pred(w1, w2, b, x) == y for x, y in XOR.items()) for w1, w2, b in itertools.product(grid, grid, grid))   # 格點搜尋:沒有一條直線分得開
# 代數矛盾:b<=0、w1+b>0、w2+b>0 ⇒ w1+w2+b = (w1+w2+2b) - b > 0,但 XOR 要求 w1+w2+b<=0
for w1, w2, b in itertools.product(grid, grid, grid):
    c = (b <= 0, w1 + b > 0, w2 + b > 0, w1 + w2 + b <= 0)
    assert not all(c)
x, y, w, b, eta = (1.0, 0.0), 1, (0.2, 0.0), 0.0, 0.1
a = w[0] * x[0] + w[1] * x[1] + b
assert abs(a - 0.2) < 1e-12 and y * a > 0                       # 感知機已分對,不更新
err = y - a; assert abs(err - 0.8) < 1e-12
dw = tuple(eta * err * xi for xi in x); db = eta * err
assert abs(dw[0] - 0.08) < 1e-12 and dw[1] == 0 and abs(db - 0.08) < 1e-12
neww = tuple(wi + d for wi, d in zip(w, dw)); newb = b + db
assert abs(neww[0] - 0.28) < 1e-12 and neww[1] == 0 and abs(newb - 0.08) < 1e-12
