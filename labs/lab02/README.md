# Lab 02:手寫反向傳播與梯度檢查

## 目標

不依賴自動微分,以 NumPy 推導並實作神經網路的反向傳播,並用**數值梯度**驗證自己的推導。

## 任務

於 `starter/lab02.py` 完成:

1. `linear_forward` / `linear_backward`: $o = xW + b$ 及其梯度
2. `relu_forward` / `relu_backward`
3. `softmax_cross_entropy`:損失 $L = -\frac{1}{N}\sum_n \log p_{n,y_n}$ 與梯度 $\frac{\partial L}{\partial z} = \frac{p - \mathbb{1}_y}{N}$

## 梯度檢查(Gradient Check)

測試以中央差分近似梯度:

$$
\frac{\partial f}{\partial x_i} \approx \frac{f(x + \epsilon e_i) - f(x - \epsilon e_i)}{2\epsilon}
$$

若你的解析梯度與數值梯度不一致,幾乎都是推導或形狀(shape)出錯。

## 評分表

| 測試 | 配分 |
|---|---|
| `test_linear_forward` | 8 |
| `test_linear_backward_shapes` | 8 |
| `test_linear_backward_gradcheck`(x、W、b 各 1 項) | 24 |
| `test_relu_forward` | 6 |
| `test_relu_backward_gradcheck` | 12 |
| `test_softmax_ce_loss_value` | 8 |
| `test_softmax_ce_gradcheck` | 14 |
| `test_softmax_ce_numerical_stability` | 8 |
| `test_two_layer_training_decreases_loss` | 12 |

## 執行

```bash
pytest labs/lab02/tests -v
```
