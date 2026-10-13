"""Lab 02 起始程式碼:手寫反向傳播(NumPy)。

請完成下列函式,使 ``labs/lab02/tests`` 全數通過。測試會用「數值梯度」
(有限差分)檢查你推導的解析梯度是否正確。

符號約定(batch 大小 $N$,輸入維度 $D$,輸出維度 $M$):
    x: (N, D)    W: (D, M)    b: (M,)    out = xW + b: (N, M)
"""

import numpy as np


def linear_forward(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """線性層前向傳播:$\\text{out} = xW + b$。"""
    # TODO
    raise NotImplementedError("請完成 linear_forward")


def linear_backward(dout: np.ndarray, x: np.ndarray, W: np.ndarray):
    """線性層反向傳播。

    Args:
        dout: 損失對輸出的梯度 $\\partial L/\\partial \\text{out}$,形狀 (N, M)。
        x: 前向傳播時的輸入 (N, D)。
        W: 權重 (D, M)。

    Returns:
        (dx, dW, db),形狀分別與 x、W、b 相同。
    """
    # TODO(提示:自己用連鎖律推導,並確認每個梯度的形狀)
    raise NotImplementedError("請完成 linear_backward")


def relu_forward(x: np.ndarray) -> np.ndarray:
    """ReLU:$\\max(0, x)$。"""
    # TODO
    raise NotImplementedError("請完成 relu_forward")


def relu_backward(dout: np.ndarray, x: np.ndarray) -> np.ndarray:
    """ReLU 反向傳播。"""
    # TODO
    raise NotImplementedError("請完成 relu_backward")


def softmax_cross_entropy(logits: np.ndarray, y: np.ndarray):
    """Softmax + 交叉熵損失(對 batch 取平均),並回傳對 logits 的梯度。

    必須對大 logits(例如 1000)保持數值穩定。

    Args:
        logits: (N, C)
        y: (N,) 整數類別標籤

    Returns:
        (loss: float, dlogits: (N, C))
    """
    # TODO(提示:softmax 前先減去每列最大值)
    raise NotImplementedError("請完成 softmax_cross_entropy")
