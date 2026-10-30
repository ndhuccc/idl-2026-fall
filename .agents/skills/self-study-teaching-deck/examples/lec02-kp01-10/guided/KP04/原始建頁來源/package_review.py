from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,shutil,zipfile
R=Path(__file__).parent;O=R/'output';now=datetime.now(ZoneInfo('Asia/Taipei')).strftime('%Y-%m-%d %H:%M')
report=f'''# IDL 第二章 KP04 圖解簡報審閱稿

完成查核：{now}（Asia/Taipei）。本機草稿，尚未發布或推送，未修改原講義及 KP03。

## 來源與格式

- 課程：ndhuccc/idl-2026-fall，main `75f6a2f8fcfdcd82f974c36a91885046a02307b4`。
- 講義：`lectures/lec02-kp04-perceptron-limits-and-generalization.md`，blob `510807905fcdaf9ab41409a87f7fa41ef3406875`。
- 實驗來源：`assets/figures/src/lec02-kp0204-figures.py`；資料與訓練函式依原始程式重現，未另抽樣替代。
- 使用課程 page-to-teaching-diagram → prompt-to-jbd-page 原生流程。已讀最新技能新增的連接線與泡泡稽核；使用本版本品質工具。HTML 轉換器與前次重新載入版本相同。
- 中英文各 21 頁、64 步（含根步驟）、1425 個可編輯原生節點，每頁三拍。無圖片節點。兩語位置、鏡頭、頁數與步驟相同。
- 主線 1–14 頁約 15–20 分鐘，第 6 頁為可跳讀的幾何直觀；15–21 頁為練習與自我檢核，不附答案。

## 內容對照

| 講義範圍 | 頁次 | 視覺教學主軸 |
|---|---|---|
| 單元主張與 §1 XOR | 1–3 | 兩道關卡、對角類別、兩條失敗邊界 |
| §1 實跑與數學限制 | 4–7 | 更新長條、四狀態循環、相交中點、有限搜尋 |
| §2 邊界與泛化 | 8–11 | 同批 24 點的兩條邊界、新點、完整分數、未見資料 |
| §3 歷史與解法 | 12–14 | 兩種限制分流、時間軸、引導 KP05 |
| 三題練習 | 15–17 | XNOR、AND 對照、未見資料證據 |
| 四題自我檢核 | 18–21 | 四選項等權重圖示與原題提示；第 1、4 題釐清題幹 |

## 活潑視覺與操作

固定「小感知器」延續 KP03 配色；本單元回到 KP02 的零一標籤，角色牌面改為 0／1。不同概念配合直尺、直角尺、旋鈕、考卷、開關、警示與抓頭表情。這些是教學比喻，不是額外實驗資料。

每頁三拍：條件與主圖 → 證據與角色提示 → 結論。第 5 頁的暫時問題在第二拍退場；第 4、10 頁第二拍以鏡頭推近，第三拍回到全景。全稿 17 條原生 connector 支援循環、分流、網路與對應。旋轉貼紙、平滑角色線條、半透明分類區、線性／徑向／多色漸層、陰影與行內關鍵字共同服務教學，功能檢查涵蓋 12/12 群。

離線 HTML：右方向鍵／空白鍵逐拍前進，左方向鍵回看，N 看備註，F 全螢幕。`漸進出場示例.gif` 是第 5 頁實際鍵盤操作擷取的三拍示範。同名 `.jbd.json` 可編輯，建頁程式也保留在工作目錄；若日後重建，應同步修改建頁腳本。

## 數值核對與講義釐清

1. XOR：零起點、學習率一、固定順序、零分判一，十輪更新數為 3、3、4、4、4、4、4、4、4、4。第四輪四次更新後回到起點，並非整輪沒有更新。有限十輪只示範行為，不能代替不可分證明。
2. 搜尋：三個整數權重各取負三至三，共 343 組，最多正確 3/4。頁面明示有限格點不是所有實數權重的證明。
3. 原資料種子 3，每類 12 點；訓練種子 100、102。兩條線皆對 24/24，對新點 (2.5,2.4) 的完整精度分數分別為 2.4643865597689807、−0.41484816157766513，輸出 1、0。
4. **原圖說錯誤**：講義稱第二條嘗試線錯兩筆。原程式為兩條線 x1+x2=0.5、x1+x2=1.5；沿用上方預測一，逐點核對分別錯一筆、三筆。簡報第 3 頁明示方向與紅圈，改為一錯／三錯；原講義未更動。
5. **泛化釐清**：新點真實標籤未提供，兩條線意見不同僅示範邊界不唯一，未測得誰泛化較好。連續 LMS 損失是另一種選擇邊界的目標，不保證較佳泛化，也不讓線性 ADALINE 解出 XOR。沒有沿用「連續損失必使邊界不貼近資料」的無條件說法。
6. **題意釐清**：檢核第 1 題改問「收斂保證」的條件，而非不可分資料連演算法都不能用；第 4 題改問 ADALINE 學習目標，而非泛化改善保證。保留選項與提示，不附答案。
7. 歷史只保留三個年份與方法，避免「第一個神經元」與「一本書單獨造成寒冬」的過度概括。

## 驗證證據

- 三輪實際瀏覽器渲染與目視檢查：第一輪 42 頁發現兩處英文溢出，另移除封面不恰當的因果箭頭；第二輪零溢出，補選項小圖與二元標籤；第三輪補長條項目及零基準、NOT 輸入輸出，最終 42 頁零溢出候選、零頁面錯誤。全部最終縮圖索引均已檢視。
- 中英文各在 1280×720、1920×1080 驗證，共 256 個步驟；進場／退場節點、連接線及標籤可見性通過，零 KaTeX、JS 錯誤與對外 HTTP 請求。見 `browser-verification.json`。
- 最新 `check_deck_quality.py --strict` 與 `check_slides.py` 兩語均通過。1319 個隱藏節點各揭示一次，泡泡文字同步、沒有空泡泡、空文字或超出頁面；字級至少 16，幾何／鏡頭對齊。見 `native-validation.json`、`structure-verification.json`。
- 提醒已核對：每語 63 個教學步驟有備註（下載會包含）；備註不放練習答案。工具另列 21 條疑似手畫箭頭；逐項核對為座標軸、類別線段、決策邊界及選項曲線靠近三角資料點的幾何判定，實際流程均為原生 connector，詳見 `箭頭提示人工核對.json`。
- `數值驗算.json`、`圖解提示詞.md`、`圖解證據表.json` 與 `逐頁排版檢查.json` 附於審閱包；`預覽截圖/` 包含完整 42 頁與 10 張雙語索引。

限制：已驗證離線 HTML 與原生檔案規則，尚未在實際 JianBauDau App 內播放或匯出 PDF。本工作區只有必要來源與工具副本，未執行完整課程 doctor／網站建置。
'''
(O/'審閱說明.md').write_text(report)
for f in ('數值驗算.json','圖解提示詞.md','圖解證據表.json'):shutil.copy2(R/f,O/f)
shutil.copy2(R/'round3/browser-audit.json',O/'逐頁排版檢查.json')
shots=O/'預覽截圖';shots.mkdir(exist_ok=True)
for p in (R/'round3').glob('*.png'):shutil.copy2(p,shots/p.name)
(R/'COURSE_STATE更新草稿.md').write_text(f'# COURSE_STATE 更新草稿（未套用至雲端）\n\n{now}（Asia/Taipei）：依教授指示完成第二章 KP04 中英文圖解簡報本機審閱稿，各21頁、64步、每頁三拍，主線14頁與練習檢核7頁。最新品質工具12/12功能涵蓋、17原生連接線、三輪渲染與256步離線播放驗證通過。圖說錯分數、泛化與檢核題意釐清詳見審閱說明。未發布、未修改原講義或KP03。輸出主機：Codex workspace；路徑 /workspace/scratch/idl-lec02-kp04-review/output/。\n')
log=Path('/workspace/my-jarvis/logs/2026-10-30-idl-lec02-kp04-圖解簡報審閱稿.md');log.parent.mkdir(parents=True,exist_ok=True)
log.write_text(f'''# IDL 第二章 KP04 圖解簡報審閱稿

- 執行者：Codex；教授指示：繼續 KP04，依重新載入的課程技能。
- 狀態：完成本機草稿，待教授審閱；完成查核時間 {now}（Asia/Taipei）。起始時間未另記錄，不估算執行時長。
- 輸入：GitHub ndhuccc/idl-2026-fall main 75f6a2f8fcfdcd82f974c36a91885046a02307b4；lectures/lec02-kp04-perceptron-limits-and-generalization.md；assets/figures/src/lec02-kp0204-figures.py。
- 輸出主機：Codex workspace；完整路徑 /workspace/scratch/idl-lec02-kp04-review/output/。
- 交付：中英文各21頁JBD／離線HTML、審閱說明、提示詞、數值與品質證據、三拍GIF與審閱包。
- 問題及補救：修正兩處英文溢出、封面誤示因果、角色二元標籤及長條解讀；原圖說第二條線錯兩筆，實算應三筆，在草稿釐清但原講義未更動。泛化與ADALINE保證、檢核第1與4題題意均列報。
- 驗證：三輪實際渲染，42頁最終零溢出；四種語言／視窗組合256步進退場與離線驗證；數值assert、原生結構與兩課程檢查通過，功能12/12、17connector。63個備註與21條幾何啟發式提醒已核對並記錄。
- 未執行：實際JianBauDau App／PDF、完整repo doctor／網站建置。未發布、未推送、未改KP03。
- 下一步：教授審閱 KP04；僅依後續指定繼續KP05。
''')
z=O/'IDL第二章KP04-審閱包.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
 for p in sorted(O.rglob('*')):
  if p.is_file() and p!=z:f.write(p,p.relative_to(O))
with zipfile.ZipFile(z) as f:assert f.testzip() is None;print('ZIP integrity PASS;',len(f.namelist()),'files;',z.stat().st_size,'bytes')
print('Packaged:',z)
