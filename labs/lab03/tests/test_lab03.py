"""Lab 03 自動評分。

需要 PyTorch;若環境未安裝(例如 CI 的輕量環境),整份測試會被略過而非失敗。
安裝:``pip install -r requirements-dl.txt``
"""

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402


def make_xor_data(n: int = 256, seed: int = 0):
    """XOR 型二分類資料:兩維同號為類別 1。線性模型無法分開,需要隱藏層。"""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=g)
    y = ((x[:, 0] * x[:, 1]) > 0).long()
    return x, y


def test_mlp_structure(lab):
    """[2,16,8,2] 應有 3 個 Linear 與 2 個 ReLU,且最後一層不是 ReLU。"""
    model = lab.MLP([2, 16, 8, 2])
    mods = list(model.modules())
    assert sum(isinstance(m, nn.Linear) for m in mods) == 3
    assert sum(isinstance(m, nn.ReLU) for m in mods) == 2
    last = [m for m in mods if isinstance(m, (nn.Linear, nn.ReLU))][-1]
    assert isinstance(last, nn.Linear)


def test_mlp_forward_shape(lab):
    out = lab.MLP([2, 16, 3])(torch.randn(5, 2))
    assert out.shape == (5, 3)


def test_mlp_parameter_count(lab):
    """[2,4,3]:(2*4+4) + (4*3+3) = 27 個參數。"""
    n = sum(p.numel() for p in lab.MLP([2, 4, 3]).parameters())
    assert n == 27


def test_train_step_returns_float_and_updates(lab):
    torch.manual_seed(0)
    model = lab.MLP([2, 8, 2])
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x, y = make_xor_data(32)
    before = [p.detach().clone() for p in model.parameters()]
    loss = lab.train_step(model, opt, x, y)
    assert isinstance(loss, float)
    after = list(model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), "參數未更新"


def test_train_step_clears_old_gradients(lab):
    """連續兩次 train_step 的梯度不應累加(必須呼叫 zero_grad)。"""
    torch.manual_seed(0)
    model = lab.MLP([2, 8, 2])
    opt = torch.optim.SGD(model.parameters(), lr=0.0)  # lr=0:參數不變,梯度應完全相同
    x, y = make_xor_data(32)
    lab.train_step(model, opt, x, y)
    g1 = [p.grad.clone() for p in model.parameters()]
    lab.train_step(model, opt, x, y)
    g2 = [p.grad.clone() for p in model.parameters()]
    for a, b in zip(g1, g2):
        torch.testing.assert_close(a, b)


def test_training_solves_xor(lab):
    """訓練後在 XOR 資料上準確率應 > 95%。"""
    torch.manual_seed(0)
    model = lab.MLP([2, 16, 2])
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    x, y = make_xor_data(256)
    for _ in range(300):
        lab.train_step(model, opt, x, y)
    assert lab.accuracy(model, x, y) > 0.95


def test_accuracy_value(lab):
    """以固定權重驗證 accuracy 的計算:恆輸出類別 1 的模型,準確率 = 類別 1 的比例。"""

    class AlwaysOne(nn.Module):
        def forward(self, x):
            out = torch.zeros(x.shape[0], 2)
            out[:, 1] = 1.0
            return out

    y = torch.tensor([1, 1, 0, 1])
    assert lab.accuracy(AlwaysOne(), torch.zeros(4, 2), y) == pytest.approx(0.75)


def test_accuracy_uses_eval_mode_and_no_grad(lab):
    """accuracy 的前向傳播必須在 eval 模式、且關閉梯度追蹤下執行。

    用探針模型記錄前向傳播當下的狀態;單純檢查 ``p.grad`` 無法偵測漏掉 no_grad。
    """

    class Probe(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 2)
            self.seen = []

        def forward(self, x):
            self.seen.append((self.training, torch.is_grad_enabled()))
            return self.fc(x)

    probe = Probe()
    probe.train()  # 故意從 train 模式開始,學生的實作必須自行切換
    x, y = make_xor_data(16)
    lab.accuracy(probe, x, y)
    assert probe.seen, "accuracy 沒有呼叫 model(x)"
    assert all(training is False for training, _ in probe.seen), "未切換為 eval 模式"
    assert all(grad is False for _, grad in probe.seen), "未使用 torch.no_grad()"
