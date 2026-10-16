# 課程大綱(Syllabus)

## 基本資訊

| 項目 | 內容 |
|---|---|
| 課程名稱 | 深度學習導論 |
| 課號 | CSIE3544AA |
| 學期 | 2026 秋季 |
| 學分 / 時數 | 3 學分 / 3 小時 |
| 上課時間地點 | 每週四下午,工A337 |
| 授課教師 | 江政欽 教授(ccchiang@gms.ndhu.edu.tw) |
| 助教 | 顏榮興、李哲瑋 |
| Office Hour | 週一 10:00–11:00(Mon. 10:00–11:00 AM) |

## 課程目標

培養學生深度學習技術的理論與實務應用能力,作為日後在機器學習與人工智慧領域研究與開發的專業基礎。

*To teach students the theory and practical application ability related to deep learning technologies as the professional foundation for their future research and development in machine learning and artificial intelligence.*

## 教科書與參考資料

- 主要教材:Ian Goodfellow, Yoshua Bengio, Aaron Courville, Deep Learning, MIT Press, 2016. ISBN: 978-0-262-03561-3
- 其他參考:見 [`references/`](references/)

## 評分方式

依教學計畫表(115 學年第 1 學期)登記的比例:

| 項目 | 比例 | 說明 |
|---|---|---|
| 平時成績(含出缺席與課堂參與) | 20% | |
| 期中考 | 25% | 第 9 週 |
| 期末考 | 15% | 第 17 週(12/31) |
| 期末專題 | 20% | 以影片與報告繳交 |
| 作業 | 20% | 3 次程式作業 |

總成績計算:

$$
S = 0.20\,S_{\text{part}} + 0.25\,S_{\text{mid}} + 0.15\,S_{\text{final}} + 0.20\,S_{\text{proj}} + 0.20\,S_{\text{hw}}
$$

教學計畫表登記的「期末考與專題」合計 35%,本課程分配為期末考 15%、期末專題 20%。

> 實際比例以第一週公告為準,任何異動記錄於 [`CHANGELOG.md`](CHANGELOG.md)。

### 作業

| 作業 | 主題 | 期間 |
|---|---|---|
| HW 01 | Backpropagation：以 NumPy 手寫兩層與三層 MLP、梯度檢查，完成非線性函數逼近及雙螺旋分類 | 2026-10-19 → 2026-11-09 23:59（台北時間） |
| HW 02 | CNN 影像分類:用 PyTorch 在 CIFAR-10 訓練 CNN,並比較至少兩種正則化策略 | 第 8 → 11 週 |
| HW 03 | 序列與注意力:用 LSTM/GRU 或簡化的自注意力處理序列任務,附簡短書面說明 | 第 12 → 15 週 |

HW 01 日期已確認；其餘作業的確切截止日於發佈時公告。

## 作業繳交規範

1. 以個人作業 Repo 繳交(**私人 Repo,必須維持 Private**,邀請教師為協作者),截止前打標籤 `hwNN-submit`,**以 GitHub 紀錄的標籤推送時間為準**;流程見 [作業說明](assignments/README.md)
2. 遲交扣分:每遲交一天扣 10%,最多扣至 50%
3. 繳交內容須能在課程指定環境下重現
4. 成績與解答**不會**公開於本 Repo
5. 截止後有課堂**作業理解測驗**(題目取自學生自己的程式),用以校準作業分數,見 [作業說明](assignments/README.md)

## 期末專題

- **分組**:每組建議 3–4 人(1–2 人亦可)。題目可從課程任一單元(必修或選修)選定;完全超出課程範圍的題目不建議。
- **期間**:第 10 週起至第 17 週;確切截止日另行公告。
- **影片**:8–12 分鐘,上傳到 YouTube 並設為「不公開」。結構為動機 → 方法 → 結果/示範 → 結論。影片連結繳交到 e 學苑。
- **報告**:寫在 GitHub 的專題 Repo(例如 `report.md`)。
- **Repo 規則**:專題 Repo 與個人作業 Repo **分開**。個人作業 Repo 維持 Private;專題 Repo **繳交時才設為 Public**。截止後仍是 Private,視為未繳交,不接受補交例外。
- 第 17 週課堂**只有期末考,不做現場發表**;同學的專題影片與 Repo 於成績結算後供觀摩。

細節見 [`projects/README.md`](projects/README.md)。

## 學術誠信

- 允許討論想法,但**程式碼與報告須獨立完成**
- 引用他人程式碼、模型或生成式 AI 輔助,須於作業中**明確標註來源與用途**
- 抄襲或代寫依校規處理

## 生成式 AI 使用政策

- 所有作業與期末專題**可以使用任何 AI 工具**。
- 唯一不可退讓的是:**你自己的思考必須在 Repo 裡看得見**。
- 每次有 AI 協助的 commit,訊息要依下列格式寫(以下為示範):

  ```
  fix: 修正輸出層反向傳播的符號錯誤
  AI-Tool: ChatGPT (GPT-4o)
  AI-Prompt: "我的梯度和數值梯度不一致,請檢查輸出層的反向傳播"
  AI-Reason: 數值梯度檢查沒通過,想找出錯誤位置
  Human-Judgment: 我自己重新推導了 dL/dz,確認 AI 指出的符號錯誤成立後才修改
  ```

- 請**多次小量提交**,不要在截止前一次大量 commit。
- AI agent 可以代為撰寫並推送 commit,但**必須先把完整訊息給你確認**。
- 教師可能隨機請你口頭說明任何一個 commit。
- 每份 `report.md` 末尾仍須有「AI 使用聲明與引用」(見 [作業說明](assignments/README.md))。

## 課堂規範

- 尊重他人,討論區禁止人身攻擊
- 請勿在公開 Issue / Discussion 張貼個資、學號、成績
