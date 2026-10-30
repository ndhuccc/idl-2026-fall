from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,shutil,zipfile
R=Path(__file__).parent;O=R/'output';now=datetime.now(ZoneInfo('Asia/Taipei')).strftime('%Y-%m-%d %H:%M')
report=f'''# IDL 第二章 KP05 圖解簡報審閱稿

完成查核：{now}（Asia/Taipei）。本機審閱草稿，未發布、未推送，未修改原講義與前面 KP。

## 技能更新與來源

先讀取 GitHub main 最新版本 `5a6e17c42083153c62d4dcce1673d86eba8d9469`，重新盤點 14 項技能，再核對本次相關四項技能與全部 28 個支援檔案的 Git blob SHA。最新副本保存在本工作目錄 `toolchain/`；來源清單見 `skills-source-manifest.json`。這是從 GitHub 更新本次技能副本，沒有反向修改 GitHub 技能。

- 使用課程 page-to-teaching-diagram、prompt-to-jbd-page、image-to-jbd-page 原生建頁庫及 slides-to-html，包含原生連接線、活潑效果、功能涵蓋、泡泡同步與 reveal 稽核。
- 講義：`lectures/lec02-kp05-mlp-and-nonlinearity.md`，blob `bdfa491ea1168d6228a943f9c6353766270c3128`。
- 圖程式：`assets/figures/src/lec02-kp0507-figures.py`，blob `a4102512bc3bfa1bdae68db7796a78c5cb277d9f`。
- 中英文各 24 頁、73 步（含根步驟）、1314 個可編輯原生節點。三拍教學，無圖片節點；雙語幾何、鏡頭及步驟一致。

## 講義對照與教學節奏

| 講義範圍 | 頁次 | 圖解 |
|---|---|---|
| 單元主張與 §1 結構 | 1–3 | 角色、全連接、前饋、六條權重線 |
| §1 神經元與矩陣 | 4–5 | 加權再活化、矩陣維度與偏置 |
| §2 手工 XOR | 6–10 | 六條權重與三偏置、四種輸入完整代入 |
| §2 隱藏座標 | 11 | 四筆映到三位置、兩筆疊合、AND 邊界 |
| §3 全線性與仿射合併 | 12–14 | 特殊抵消、一般偏置合併、多層矩陣連乘 |
| §3 原種子實驗 | 15 | 相同權重下恆等／ReLU 數值等值線 |
| 迷思、選讀、下一 KP | 16–17 | 表示與學習分開、階梯的梯度問題 |
| 三題練習 | 18–20 | 重算前饋、合併矩陣、十層恆等 |
| 四題自我檢核 | 21–24 | 前饋、層數約定、無偏置全線性、非線性 |

主線 17 頁適合約 20 分鐘講授；練習檢核 7 頁可另行操作。主線保留講義四種完整手算例；延伸練習頁及其講者備註只給提示，不另放練習或選項答案。第 22 頁題意經修訂，詳見下節。

## 視覺與播放

每頁延續藍頭紫身橘天線「小感知器」，依內容換開關、旋鈕、直尺、警示或問號牌。固定顏色區分輸入、隱藏與輸出；數值與座標圖由程式產生，角色及道具標為教學比喻。另含旋轉貼紙、平滑角色線條、徑向光暈、陰影、半透明圖框、多色標題與行內重點。

全稿 52 條原生 connector 用於神經網路、運算流程與選項示意。手算頁第一拍先看輸入與 OR，第二拍展開 NAND、AND 和角色；第 15 頁第二拍才出現 ReLU 圖。第 14 頁問題提示在第二拍退場，第 13 頁推近聚焦後回全景。`漸進出場示例.gif` 為第 14 頁實際鍵盤擷取的三拍。

離線 HTML 用右方向鍵／空白鍵逐拍前進、左方向鍵回看、N 看備註、F 全螢幕。同名 JBD 是可編輯來源；重建時應同步修改 `build_kp05.py` 等建頁程式，避免手改 JSON 被重建覆蓋。

## 數值與內容釐清

- XOR 沿用講義手工權重：OR 為 (1,1)、偏置 −1；NAND 為 (−1,−1)、偏置 1.5；AND 為 (1,1)、偏置 −2。階梯在零點輸出一。四種輸入輸出精確為 0、1、1、0，完整中間分數與活化均已 assert。這是講義明示補充的構造，不是訓練成果。
- 明確修正「隱藏層看不到原始資料」：第一個隱藏層直接接收輸入，更深層接前層特徵；隱藏是結構角色，不表示無資料或無監督訊號。
- 原講義混用 W 轉置及不轉置的寫法。本稿統一 A 為輸出維度乘輸入維度，欄向量輸入輸出，並明示 A_k=W_k 的轉置；偏置顯式列出。含偏置合併為 A2 A1 與 A2 b1+b2。已用非定常三層例子數值驗證一般仿射合併。
- 同一組 XOR 權重改恆等活化時，A2 A1=(0,0)，偏置合併 −1+1.5−2=−1.5，故這個特例為常數。一般全線性網路只是線性／仿射映射，不必是常數；也不沿用「連 XOR 一半都做不到」的措辭：若此常數再以零為門檻判類零，可對四點中的兩點。
- 原種子 2 實驗重現 2→4→3→1 網路，固定所有權重、無偏置、只切換兩個隱藏活化，輸出仍線性。原程式用 300×300 網格與 1e−9 等值門檻；恆等網路與合併矩陣的差異小於 1e−9。ReLU 有 32567／90000 個網格點恰為零，且數值零平台可能產生碎小等值線。本稿依原程式保留超過 40 頂點的主要等值線，以原生折線畫出，不填色或宣稱精確四個分類區域，不把數值門檻冒充一般精確零分界；這不是泛化或訓練比較。
- 檢核第 2 題原四個選項沒有合理正解，且將輸入層混稱隱藏層。本稿標示「原題釐清」，改問閱讀層數時應先確認的計數約定，並修訂相應選項。第 3 題加「無偏置」條件，讓「一次線性映射」與前文含偏置仿射情況清楚區分。
- 非線性提供表示 XOR 的可能性，不保證容量足夠、權重正確或訓練一定成功；表示、參數化與學習行為分開討論。

以上只在審閱稿釐清，原講義未更動。原始歷史取捨沿用講義，不新增單一年份發明權的斷言。

## 驗證

- 三輪實際渲染與目視檢查：第一輪核對全部 48 頁，修正隱藏座標對應、抵消長條標示、階梯軸標與檢核箭頭；第二輪檢視重點頁並調整分拍順序；第三輪再檢視全部雙語索引，48 頁零文字溢出候選、零頁面錯誤。
- 最新 `check_deck_quality.py --strict` 與課程 `check_slides.py` 兩語通過。功能涵蓋 11/12 群，高於門檻 9/12；未為湊齊形狀類別額外加入無教學用途的星形。每頁至少三類活潑效果、1193 個隱藏節點各揭示一次，泡泡文字同步，無空文字或超出頁面。
- 中英文 × 1280×720／1920×1080 共四組、292 個步驟逐步驗證進場／退場、連接線與標籤可見性，零 KaTeX 與 JS 錯誤、零外部 HTTP 請求。見 `browser-verification.json`。
- 原生格式、場景與連接線引用、字級至少 16、雙語幾何與鏡頭一致均通過。見 `structure-verification.json`、`native-validation.json`。
- 已核對提醒：每語 72 個教學步驟含備註，下載會包含；3 條疑似手畫箭頭實為第 11 頁兩組座標軸與 AND 邊界靠近三角資料點的啟發式提示，實際流程全用 connector，見 `箭頭提示人工核對.json`。
- 審閱包附數值驗算、逐頁提示詞／證據表、技能版本清單、最終 48 張完整截圖與 10 張索引。

限制：驗證涵蓋離線 HTML 與 JBD 資料規則，尚未在實際 JianBauDau App 播放或匯出 PDF；本機只有必要工具副本，未執行完整課程 doctor／網站建置。
'''
(O/'審閱說明.md').write_text(report)
for f in ('數值驗算.json','圖解提示詞.md','圖解證據表.json','skills-source-manifest.json'):shutil.copy2(R/f,O/f)
shutil.copy2(R/'round3/browser-audit.json',O/'逐頁排版檢查.json');shots=O/'預覽截圖';shots.mkdir(exist_ok=True)
for p in (R/'round3').glob('*.png'):shutil.copy2(p,shots/p.name)
(R/'COURSE_STATE更新草稿.md').write_text(f'# COURSE_STATE 更新草稿（未同步雲端）\n\n{now}（Asia/Taipei）：依教授指示先核對 GitHub 最新5a6e17c，更新相關四項技能28檔，再完成第二章KP05中英文圖解審閱稿，各24頁、73步、三拍。17頁主線與7頁練習檢核，52條原生連接線。三輪渲染、數值驗算、原生檢查與292步離線進退場驗證通過；功能11/12。隱藏層、全線性特例、矩陣約定、ReLU數值等值線及檢核題意的釐清詳見審閱說明。未發布、未改原講義或前KP。輸出主機：Codex workspace；完整路徑 /workspace/scratch/idl-lec02-kp05-review/output/。\n')
log=Path('/workspace/my-jarvis/logs/2026-10-30-idl-lec02-kp05-圖解簡報審閱稿.md');log.parent.mkdir(parents=True,exist_ok=True)
log.write_text(f'''# IDL 第二章 KP05 審閱稿

- 執行者：Codex。教授指示：先更新GitHub最新版技能，再繼續KP05。
- 狀態：完成本機審閱稿；查核時間 {now}（Asia/Taipei）。起始時間未另記錄，不估算時長。
- 技能：GitHub ndhuccc/idl-2026-fall main 5a6e17c42083153c62d4dcce1673d86eba8d9469；盤點14項，相關四項技能及28檔SHA核對後更新本機副本，未反向改GitHub。
- 來源：lectures/lec02-kp05-mlp-and-nonlinearity.md；assets/figures/src/lec02-kp0507-figures.py。
- 輸出主機：Codex workspace；完整路徑 /workspace/scratch/idl-lec02-kp05-review/output/。
- 成果：中英文各24頁、73步、1314節點，52原生連接線；JBD／離線HTML／GIF／審閱說明／驗證證據／ZIP。
- 問題與處理：矩陣記號混用改為明確A=W轉置；第一hidden看rawinput、特殊線性抵消非通則；原檢核第2題無合理答案，明示修題；ReLU零平台與1e−9等值線限制明示。原講義不變。
- 驗證：三輪實際渲染48頁零溢出、數值assert、雙語原生結構、課程品質strict通過，功能11/12；292步離線進退場／連線驗證通過。72備註與3條坐標線啟發式提醒已核對。
- 限制：實際JBD App與PDF、完整課程doctor／網站建置未執行；未發布、未推送、未改前KP。
- 下一步：教授審閱KP05，後續依指定繼續KP06。
''')
z=O/'IDL第二章KP05-審閱包.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
 for p in sorted(O.rglob('*')):
  if p.is_file() and p!=z:f.write(p,p.relative_to(O))
with zipfile.ZipFile(z) as f:assert f.testzip() is None;print('ZIP integrity PASS;',len(f.namelist()),'files;',z.stat().st_size,'bytes')
