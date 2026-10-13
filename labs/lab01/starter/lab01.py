"""Lab 01 起始程式碼(Starter)。

請完成下列三個函式,使 ``labs/lab01/tests`` 中的測試全部通過。
不要更改函式名稱與參數,否則自動評分會失敗。

數學定義:
    sigmoid:  $\\sigma(z) = \\frac{1}{1 + e^{-z}}$
    mse:      $\\frac{1}{n}\\sum_i (\\hat{y}_i - y_i)^2$
    gd_step:  $\\theta \\leftarrow \\theta - \\eta \\nabla_\\theta L$
"""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """逐元素計算 sigmoid 函數。

    Args:
        z: 任意形狀的陣列。

    Returns:
        與 ``z`` 同形狀的陣列,值域為 (0, 1)。
        需對很大的負值(例如 -1000)保持數值穩定,不可產生溢位警告。
    """
    # TODO: 請實作(提示:對 z >= 0 與 z < 0 分開處理可避免 exp 溢位)
    raise NotImplementedError("請完成 sigmoid")


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """計算均方誤差(Mean Squared Error)。

    Args:
        y_pred: 預測值,形狀 (n,)。
        y_true: 真實值,形狀 (n,)。

    Returns:
        純量 float。
    """
    # TODO: 請實作
    raise NotImplementedError("請完成 mse")


def gd_step(theta: np.ndarray, grad: np.ndarray, lr: float) -> np.ndarray:
    """執行一步梯度下降更新。

    Args:
        theta: 目前參數。
        grad: 損失對參數的梯度,形狀與 ``theta`` 相同。
        lr: 學習率 $\\eta$。

    Returns:
        更新後的新參數(**不可就地修改**傳入的 ``theta``)。
    """
    # TODO: 請實作
    raise NotImplementedError("請完成 gd_step")
