---
name: view-from-text-v3
description: Turn a lecture (text) into a JianBauDau deck of native-node information graphics, designing each figure freely in HTML/CSS/SVG (KaTeX for math) and auto-converting it to native nodes (html2nfig). Plan pages first, design the figure in HTML, convert to nodes, audit, get the teacher's confirmation, publish. Data charts can still use the v2 layout library. Use when figures need richer visual design than the fixed v2 layouts.
---

# view-from-text-v3:HTML 自由設計 → 原生節點

教授指示(2026-10-26):用「規劃 → 以 HTML/CSS/SVG 自由設計資訊圖(數學用 KaTeX) → 自動轉成原生節點」的流程產生簡報。視覺設計不再受 v2 版型庫限制,成品仍是原生節點(可編輯、可翻譯、有 term 定義),講義 PNG 與簡報來自同一份設計。

v3 沿用 v1/v2 的規則與工具(CLARITY、LEVELS、NARRATIVE、`deckv2`、`audit_deck`、`preview_deck`、`plan_pages.py`);新增 `tools/htmlkit.py`(設計 helper + lint + 轉換)與單元腳手架。**先讀 `DESIGN.md`(HTML 設計規則與踩雷)。**

## 流程(每個單元 KP)

1. **盤點與規劃**(同 v2):讀講義小節;寫 `kpNN_plan.json`(每頁 title / covers / main / layout / question / takeaway / numbers / source / on_page),跑 `view-from-text-v2/tools/plan_pages.py`;圖內數字寫 `kpNN_nums.py` 重算。**規劃(含每張圖的圖簡報)交教授確認("如你建議")後才畫。**
2. **建立單元資料夾**:`python3 tools/new_unit.py <dir> KPNN "<中文單元名>" "<English unit name>"`(產生 `kpNN_deck.py`、`kpNN_text.py`、`terms*.json`、`deckcommon.py`、`W_zh/`、`W_en/`)。
3. **選畫圖路線**(每頁各選一種):
   - **A. HTML 自由設計**(概念圖、架構圖、比較圖、流程、公式推導):用 `tools/htmlkit.py` 寫 `design(lang)` 回傳 HTML 字串;`build(fid, design, 1160, 470, outdir)` 產生 `<fid>_<lang>.html/.png/.json` 並 lint;`native(fid, figid, 1160, 470, jsondir, alt, lang)` 取得 `NFig`,在 `kpNN_text.py` 以 `layout="freeform", spec=lambda: native(...)` 放進頁面。範例 `examples/kp18_pnn.py`。
   - **B. v2 版型庫**(資料圖:`plot_xy`、`bars_compare` 等):數字由程式算出,畫法見 `view-from-text-v2/LAYOUTS.md`。
   - **C. 教授提供的圖(截圖/生成圖)**:在 HTML 中重建,再轉換;不把點陣圖放進成品。
   - **D. 要視覺探索**:先用 `anthropic-skills:page-to-teaching-diagram` 產生提示詞/設計稿,再用 A 實作。
4. **建置與稽核**:`MIN=<分鐘> bash tools/run_unit.sh <dir> kpNN_deck`(fit → audit_deck → preview → bottoms)。FAIL 必須為 0;看 `W_zh/prev`、`W_en/prev` 的截圖。
5. **交教授確認**:傳中、英文各一張截圖;**教授說「發佈」才寫入 repo**。
6. **發佈**:複製 `draft.jbd.json` 到 `slides/lec05-kpNN-*.jbd.json`(英文加 `-en`);`tools/gen_figs.py` 產生講義 PNG;`tools/insert_figs.py` 更新講義(中英)圖說;更新 `CHANGELOG.md`、`COURSE_STATE.md`、`schedule.md`(若有變動)、日誌;跑 `check_slides.py` 與 `build_docs.py && mkdocs build`;提交推送 `main`;把 skills 資料夾同步到 course-template 與 idl-2026-fall。

## 新功能(2026-10-27)
填色透明度、漸層、陰影與多邊形已可用,頁面也不再保留 KP 標題區(整頁 1280×720);用法見 `DESIGN.md` 最後一節與 `docs/`。

## 規則(摘要)
- 原生節點成品;字級 ≥ 16px;只用純色(漸層取中間色、陰影會被丟棄);圖寬 1160、高 ≤ 480。
- 每頁自足:本頁符號與名詞在圖內定義;term 定義節點用 `named(..., name="term:詞")`。
- 數字由程式算出,替代文字列出所有數字與結論;英文版逐字翻譯並重跑稽核。
- 數學一律 KaTeX(`k(...)`),轉換後成為 `$…$` 數學節點;含 `\left` 的式子留足高度。
- 設計完要看 lint 輸出(字級、出界、溢出、自動換行)與截圖,再看轉換後的 preview。

## 檔案
- `tools/htmlkit.py`、`tools/new_unit.py`、`tools/run_unit.sh`、`tools/gen_figs.py`、`tools/insert_figs.py`
- 依賴:`view-from-text-v2/tools/figkit/html2nfig.py`、`view-from-text/tools/deckkit/nativefig.py`(`NFig.text(..., h=)`)、`slides-to-html/node_modules/katex`、Playwright/Chromium。
