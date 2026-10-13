"""Lab 01 自動評分測試。每個測試對應評分表中的一個項目。"""

import warnings

import numpy as np
import pytest


# ---------- sigmoid ----------
def test_sigmoid_known_values(lab):
    """sigma(0)=0.5,且與已知值吻合。"""
    z = np.array([0.0, 1.0, -1.0])
    expected = np.array([0.5, 0.7310585786300049, 0.2689414213699951])
    np.testing.assert_allclose(lab.sigmoid(z), expected, rtol=1e-9)


def test_sigmoid_numerical_stability(lab):
    """極端值不可產生溢位警告,且結果趨近 0 或 1。"""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 任何警告都視為失敗
        out = lab.sigmoid(np.array([-1000.0, 1000.0]))
    np.testing.assert_allclose(out, [0.0, 1.0], atol=1e-12)


def test_sigmoid_preserves_shape(lab):
    """輸出形狀須與輸入相同。"""
    z = np.zeros((3, 4))
    assert lab.sigmoid(z).shape == (3, 4)


# ---------- mse ----------
def test_mse_value(lab):
    """mse([1,2,3],[1,2,5]) = (0+0+4)/3。"""
    assert lab.mse(np.array([1, 2, 3]), np.array([1, 2, 5])) == pytest.approx(4 / 3)


def test_mse_zero_when_equal(lab):
    """預測等於真值時為 0。"""
    y = np.random.default_rng(0).normal(size=10)
    assert lab.mse(y, y) == pytest.approx(0.0)


# ---------- gd_step ----------
def test_gd_step_value(lab):
    """theta - lr * grad。"""
    out = lab.gd_step(np.array([1.0, 2.0]), np.array([0.5, -1.0]), lr=0.1)
    np.testing.assert_allclose(out, [0.95, 2.1])


def test_gd_step_not_inplace(lab):
    """不可就地修改輸入。"""
    theta = np.array([1.0, 2.0])
    lab.gd_step(theta, np.array([1.0, 1.0]), lr=0.5)
    np.testing.assert_allclose(theta, [1.0, 2.0])


def test_gd_minimizes_quadratic(lab):
    """對 L(x)=x^2 反覆更新,應收斂到 0。"""
    x = np.array([5.0])
    for _ in range(200):
        x = lab.gd_step(x, 2 * x, lr=0.1)
    assert abs(float(x[0])) < 1e-6
