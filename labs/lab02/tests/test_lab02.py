"""Lab 02 自動評分:以數值梯度(中央差分)驗證解析梯度。

中央差分:$\\partial f/\\partial x_i \\approx (f(x+\\epsilon e_i) - f(x-\\epsilon e_i)) / 2\\epsilon$
"""

import numpy as np
import pytest


def numerical_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """逐元素計算純量函數 f 對 x 的數值梯度(不修改傳入的 x)。"""
    x = x.astype(float).copy()
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        plus = f(x)
        x[idx] = old - eps
        minus = f(x)
        x[idx] = old
        grad[idx] = (plus - minus) / (2 * eps)
    return grad


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    return {
        "x": rng.normal(size=(4, 3)),
        "W": rng.normal(size=(3, 5)),
        "b": rng.normal(size=(5,)),
        "dout": rng.normal(size=(4, 5)),
    }


# ---------- linear ----------
def test_linear_forward(lab, data):
    out = lab.linear_forward(data["x"], data["W"], data["b"])
    np.testing.assert_allclose(out, data["x"] @ data["W"] + data["b"])


def test_linear_backward_shapes(lab, data):
    dx, dW, db = lab.linear_backward(data["dout"], data["x"], data["W"])
    assert dx.shape == data["x"].shape
    assert dW.shape == data["W"].shape
    assert db.shape == data["b"].shape


@pytest.mark.parametrize("name", ["x", "W", "b"])
def test_linear_backward_gradcheck(lab, data, name):
    """對 x、W、b 分別做梯度檢查,使用隨機的上游梯度 dout 作為權重。"""
    dout = data["dout"]

    def loss_of(**override):
        args = {**data, **override}
        return float((lab.linear_forward(args["x"], args["W"], args["b"]) * dout).sum())

    analytic = dict(zip("x W b".split(), lab.linear_backward(dout, data["x"], data["W"])))[name]
    numeric = numerical_grad(lambda v: loss_of(**{name: v}), data[name])
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


# ---------- relu ----------
def test_relu_forward(lab):
    x = np.array([[-1.0, 0.0, 2.0]])
    np.testing.assert_allclose(lab.relu_forward(x), [[0.0, 0.0, 2.0]])


def test_relu_backward_gradcheck(lab):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 4))
    x[np.abs(x) < 0.1] += 0.5  # 避開 0 附近(ReLU 在 0 不可微,數值梯度不可靠)
    dout = rng.normal(size=x.shape)
    analytic = lab.relu_backward(dout, x)
    numeric = numerical_grad(lambda v: float((lab.relu_forward(v) * dout).sum()), x)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


# ---------- softmax cross-entropy ----------
def test_softmax_ce_loss_value(lab):
    """均勻 logits、C 類時損失應為 log(C)。"""
    loss, _ = lab.softmax_cross_entropy(np.zeros((5, 4)), np.array([0, 1, 2, 3, 0]))
    assert loss == pytest.approx(np.log(4))


def test_softmax_ce_gradcheck(lab):
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(6, 5))
    y = rng.integers(0, 5, size=6)
    _, analytic = lab.softmax_cross_entropy(logits, y)
    numeric = numerical_grad(lambda v: lab.softmax_cross_entropy(v, y)[0], logits)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


def test_softmax_ce_numerical_stability(lab):
    """極大 logits 不可產生 nan / inf。"""
    logits = np.array([[1000.0, 0.0, -1000.0], [0.0, 1000.0, 0.0]])
    loss, grad = lab.softmax_cross_entropy(logits, np.array([0, 1]))
    assert np.isfinite(loss) and np.all(np.isfinite(grad))
    assert loss == pytest.approx(0.0, abs=1e-9)


def test_two_layer_training_decreases_loss(lab):
    """整合測試:用你的函式訓練 2 層網路,損失應明顯下降。"""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(64, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)  # XOR 型資料
    W1, b1 = rng.normal(size=(2, 16)) * 0.5, np.zeros(16)
    W2, b2 = rng.normal(size=(16, 2)) * 0.5, np.zeros(2)

    def step(lr=0.5):
        nonlocal W1, b1, W2, b2
        h_pre = lab.linear_forward(X, W1, b1)
        h = lab.relu_forward(h_pre)
        logits = lab.linear_forward(h, W2, b2)
        loss, dlogits = lab.softmax_cross_entropy(logits, y)
        dh, dW2, db2 = lab.linear_backward(dlogits, h, W2)
        dh_pre = lab.relu_backward(dh, h_pre)
        _, dW1, db1 = lab.linear_backward(dh_pre, X, W1)
        W1, b1, W2, b2 = W1 - lr * dW1, b1 - lr * db1, W2 - lr * dW2, b2 - lr * db2
        return loss

    first = step()
    for _ in range(300):
        last = step()
    assert last < 0.5 * first
