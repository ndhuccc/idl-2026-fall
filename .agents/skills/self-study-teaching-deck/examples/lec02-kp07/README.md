# 範例:第 2 週 KP07 損失函數(同一 KP、兩種風格)

輸入文本:`input/lec02-kp07-loss-functions.md`(講義快照,來自 repo 提交 c7142b8;約 17 KB,含公式、算例、迷思、練習、小測驗)。講義引用的兩張圖 `lec02-kp07-loss-regression`、`lec02-kp07-loss-hinge` 不在此收錄,兩種風格都用程式依數字重畫。學習「怎麼切頁」時,對照輸入文本與 `evidence-*.json` 的 `source_anchors`;覆蓋檢查請對 `input/` 的快照執行。兩種風格都**完整覆蓋講義**,共 29 頁,頁數是由內容決定的(兩邊恰好相同,是因為共用同一份切分)。

| | 圖解版 `diagram/` | 導學版 `guided/` |
|---|---|---|
| 吉祥物 | 無 | 每頁,道具與表情隨概念 |
| 圖面 | 全版圖解,公式與數字卡片 | 左 3/4 圖解,右側角色與對話框 |
| 建頁工具 | `diagram/kit/` + `kp07_diagram_pages*.py` | `guided/建頁來源/`(pagekit、mascot、specs) |
| 來源 | 本技能試行產出 | 另一位 agent 用 page-to-teaching-diagram 產出(使用者上傳) |

- `kp07_shared.py`:共用內容規格(標題、結論、講義錨點、承接小標)。
- `make_evidence.py` → `evidence-diagram.json`、`evidence-guided.json`。
- 檢查:`python3 tools/check_coverage.py <講義> examples/lec02-kp07/evidence-*.json`;`python3 tools/check_style.py --style diagram|guided <簡報> <證據表>`。
- 圖解版重建:在 `diagram/` 執行
  `PYTHONPATH=.:kit python3 kit/build_kp.py kp07_diagram_pages OUT "KP07 …" "KP07 …" lec02-kp07-loss-functions`
- `diagram/sheet_*.png` 是圖解版中文預覽拼圖。

已知限制:導學版的簡報與圖解提示詞是原作者的版本,未依本技能的「承接小標」「自學導讀稿」規則重做;`evidence-guided.json` 的承接小標是為檢查而補的,導學版畫面上尚無此行。
