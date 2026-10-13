# Lab 03:以 PyTorch 建立並訓練 MLP

## 目標

用 PyTorch 實作標準訓練流程,並理解為什麼 XOR 問題需要隱藏層與非線性激活函數。

## 任務

於 `starter/lab03.py` 完成:

1. `MLP(sizes)`:`Linear → ReLU → … → Linear`,最後一層輸出 logits
2. `train_step(model, optimizer, x, y)`:一次完整的訓練步驟,回傳 `float` 損失
3. `accuracy(model, x, y)`:評估準確率(`eval` 模式、不追蹤梯度)

訓練目標為交叉熵損失:

$$
L = -\frac{1}{N}\sum_{n=1}^{N} \log \frac{e^{z_{n,y_n}}}{\sum_c e^{z_{n,c}}}
$$

## 評分表

| 測試 | 配分 |
|---|---|
| `test_mlp_structure` | 15 |
| `test_mlp_forward_shape` | 10 |
| `test_mlp_parameter_count` | 10 |
| `test_train_step_returns_float_and_updates` | 15 |
| `test_train_step_clears_old_gradients` | 15 |
| `test_training_solves_xor` | 15 |
| `test_accuracy_value` | 10 |
| `test_accuracy_uses_eval_mode_and_no_grad` | 10 |

## 環境

```bash
pip install -r requirements-dl.txt   # 含 torch
pytest labs/lab03/tests -v
```

> 未安裝 PyTorch 時,測試會顯示 *skipped* 而不是失敗。
