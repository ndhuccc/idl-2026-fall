"""Lab 01 參考解答(Solution)。

⚠️ 此檔案不應在作業截止前公開。
公開 Repo 請搭配 ``.github/workflows/release-solutions.yml`` 於截止後才釋出,
或將 ``solution/`` 移至私有 Repo。
"""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """數值穩定的 sigmoid:$\\sigma(z) = 1 / (1 + e^{-z})$。

    對 z >= 0 使用 1 / (1 + exp(-z));對 z < 0 使用 exp(z) / (1 + exp(z)),
    兩種寫法在各自區間內 exp 的引數皆 <= 0,因此不會溢位。
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """均方誤差:$\\frac{1}{n}\\sum_i (\\hat{y}_i - y_i)^2$。"""
    diff = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.mean(diff ** 2))


def gd_step(theta: np.ndarray, grad: np.ndarray, lr: float) -> np.ndarray:
    """一步梯度下降:回傳新陣列,不修改輸入。"""
    return np.asarray(theta, dtype=float) - lr * np.asarray(grad, dtype=float)
