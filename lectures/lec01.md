# Lecture 01:課程介紹與數學預備

> 週次:第 1 週

## 學習目標

- 了解課程架構與評分
- 複習向量、矩陣與梯度的基本符號
- 完成環境設定

## 1. 符號約定

| 符號 | 意義 |
|---|---|
| $\mathbf{x} \in \mathbb{R}^d$ | 輸入向量 |
| $W \in \mathbb{R}^{m \times d}$ | 權重矩陣 |
| $L(\theta)$ | 損失函數 |

## 2. 一個最小的模型

線性模型:

$$
\hat{\mathbf{y}} = W\mathbf{x} + \mathbf{b}
$$

以梯度下降最小化損失:

$$
\theta_{t+1} = \theta_t - \eta\, \nabla_\theta L(\theta_t)
$$

## 3. 小結

- 模型 = 函數 + 參數
- 學習 = 以資料調整參數以降低損失

## 延伸閱讀

見 [`references/`](../references/README.md)。
