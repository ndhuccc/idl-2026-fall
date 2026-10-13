"""Lab 03 起始程式碼:以 PyTorch 建立並訓練 MLP。

請完成 ``MLP``、``train_step``、``accuracy``,使 ``labs/lab03/tests`` 全數通過。
"""

from typing import Sequence

import torch
from torch import nn


class MLP(nn.Module):
    """多層感知器:Linear -> ReLU -> ... -> Linear(最後一層不加激活函數)。

    Args:
        sizes: 各層維度,例如 ``[2, 16, 2]`` 表示 2 -> 16 -> 2。
            參數層數為 ``len(sizes) - 1``。
    """

    def __init__(self, sizes: Sequence[int]):
        super().__init__()
        # TODO: 建立 self.net(提示:nn.Sequential;最後一層輸出 logits,不加 ReLU)
        raise NotImplementedError("請完成 MLP.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """輸入 (N, sizes[0]),輸出 logits (N, sizes[-1])。"""
        # TODO
        raise NotImplementedError("請完成 MLP.forward")


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer,
               x: torch.Tensor, y: torch.Tensor) -> float:
    """執行一次訓練步驟並回傳該步損失(Python float)。

    順序:清除舊梯度 -> 前向 -> 損失(交叉熵)-> 反向 -> 更新參數。
    """
    # TODO
    raise NotImplementedError("請完成 train_step")


def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """計算分類準確率(0~1)。評估時應切換為 eval 模式且不追蹤梯度。"""
    # TODO
    raise NotImplementedError("請完成 accuracy")
