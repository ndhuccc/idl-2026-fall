"""Lab 03 參考解答:以 PyTorch 建立並訓練 MLP。

⚠️ 此檔案不應在作業截止前公開。
"""

from typing import Sequence

import torch
from torch import nn


class MLP(nn.Module):
    """多層感知器:Linear -> ReLU -> ... -> Linear(最後一層不加激活函數)。

    Args:
        sizes: 各層維度,例如 ``[2, 16, 2]`` 表示 2 -> 16 -> 2。
    """

    def __init__(self, sizes: Sequence[int]):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:  # 最後一層輸出 logits,不加 ReLU
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """輸入 (N, sizes[0]),輸出 logits (N, sizes[-1])。"""
        return self.net(x)


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer,
               x: torch.Tensor, y: torch.Tensor) -> float:
    """執行一次訓練步驟並回傳該步損失(Python float)。

    順序:清除舊梯度 -> 前向 -> 損失 -> 反向 -> 更新參數。
    """
    model.train()
    optimizer.zero_grad()
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """計算分類準確率(0~1)。評估時切換為 eval 模式且不追蹤梯度。"""
    model.eval()
    pred = model(x).argmax(dim=1)
    return float((pred == y).float().mean().item())
