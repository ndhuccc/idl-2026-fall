"""Lab 02 參考解答:手寫反向傳播(NumPy)。

⚠️ 此檔案不應在作業截止前公開。

符號約定(batch 大小 $N$,輸入維度 $D$,輸出維度 $M$):
    x: (N, D)    W: (D, M)    b: (M,)    out = xW + b: (N, M)
"""

import numpy as np


def linear_forward(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """線性層前向傳播:$\\text{out} = xW + b$。"""
    return x @ W + b


def linear_backward(dout: np.ndarray, x: np.ndarray, W: np.ndarray):
    """線性層反向傳播。

    由連鎖律:
        $\\partial L/\\partial x = \\text{dout}\\, W^\\top$
        $\\partial L/\\partial W = x^\\top\\, \\text{dout}$
        $\\partial L/\\partial b = \\sum_n \\text{dout}_n$

    Returns:
        (dx, dW, db),形狀分別與 x、W、b 相同。
    """
    dx = dout @ W.T
    dW = x.T @ dout
    db = dout.sum(axis=0)
    return dx, dW, db


def relu_forward(x: np.ndarray) -> np.ndarray:
    """ReLU:$\\max(0, x)$。"""
    return np.maximum(0.0, x)


def relu_backward(dout: np.ndarray, x: np.ndarray) -> np.ndarray:
    """ReLU 反向傳播:輸入 > 0 的位置讓梯度通過,其餘為 0。"""
    return dout * (x > 0)


def softmax_cross_entropy(logits: np.ndarray, y: np.ndarray):
    """Softmax + 交叉熵損失(對 batch 取平均),並回傳對 logits 的梯度。

    為數值穩定,先減去每列最大值(不改變 softmax 結果)。

    損失:$L = -\\frac{1}{N}\\sum_n \\log p_{n, y_n}$
    梯度:$\\partial L/\\partial z = (p - \\mathbb{1}_y)/N$

    Args:
        logits: (N, C)
        y: (N,) 整數類別標籤

    Returns:
        (loss: float, dlogits: (N, C))
    """
    n = logits.shape[0]
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(n), y]).mean()
    dlogits = probs.copy()
    dlogits[np.arange(n), y] -= 1.0
    dlogits /= n
    return float(loss), dlogits
