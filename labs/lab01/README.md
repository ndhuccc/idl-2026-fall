# Lab 01:數值基礎 — sigmoid、MSE 與梯度下降

## 目標

以 NumPy 實作深度學習中最基本的三個元件,並體會數值穩定性。

## 任務

於 `starter/lab01.py` 完成:

1. `sigmoid(z)`:$\sigma(z) = \dfrac{1}{1 + e^{-z}}$,須對極端值保持穩定
2. `mse(y_pred, y_true)`:$\dfrac{1}{n}\sum_{i=1}^{n} (\hat{y}_i - y_i)^2$
3. `gd_step(theta, grad, lr)`:$\theta \leftarrow \theta - \eta \nabla_\theta L$

## 評分表

| 測試 | 配分 |
|---|---|
| `test_sigmoid_known_values` | 15 |
| `test_sigmoid_numerical_stability` | 15 |
| `test_sigmoid_preserves_shape` | 10 |
| `test_mse_value` | 15 |
| `test_mse_zero_when_equal` | 10 |
| `test_gd_step_value` | 10 |
| `test_gd_step_not_inplace` | 10 |
| `test_gd_minimizes_quadratic` | 15 |

## 執行

```bash
pytest labs/lab01/tests -v
```
