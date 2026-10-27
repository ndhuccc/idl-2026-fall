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

## 環境準備(新環境第一次做,一次就好)
```bash
npm ci --ignore-scripts --prefix .agents/skills/slides-to-html     # KaTeX(node_modules 被 .gitignore 排除,必須自己裝)
pip install numpy scipy matplotlib pillow pytest                    # 數字、圖與 LP 實驗
python3 -c "import playwright"                                      # 需要 Playwright + Chromium(html2nfig 用);雲端環境已預裝
fc-list | grep -i "zen hei\|noto sans cjk"                         # 要有中文字型(圖的 PNG 用 WenQuanYi Zen Hei)
```
暫存資料夾(例如 `scratchpad/kpNN`)在對話結束後會消失;每個單元做完把 `kpNN_nums.py`、`kpNN_pages.py`、`kpNN_text.py` 放進 `examples/`(發佈時一起提交),`caps.json`、`map.json` 也保留為範本(見 `examples/kp13_caps.json`)。

## 發佈細節(教授說「發佈 KPNN」之後,逐步)
1. 複製 `W_zh/draft.jbd.json`、`W_en/draft.jbd.json` 到 `slides/lec05-kpNN-<名稱>.jbd.json` 與 `...-en.jbd.json`。
2. **講義圖**:`python3 tools/make_specs.py <單元資料夾> kpNN <頁數>` 產生 `kpNN_specs.py`;寫 `map.json`(`{"p1": "lec05-kpNN-<id>", ...}`,一頁對一個圖檔名;有合併的頁可只保留一張);再
   `python3 tools/gen_figs.py <單元資料夾>/kpNN_specs.py <單元資料夾>/map.json assets/figures`(輸出中文與 `-en` 英文版 PNG)。
3. **圖說**:寫 `caps.json`(格式見 `tools/insert_figs.py` 檔頭與 `examples/kp13_caps.json`;圖說含 `〔功能：N〕`,alt 文字含關鍵數字)。
   `python3 tools/insert_figs.py caps.json lectures/lec05-kpNN-*.md lectures/en/lec05-kpNN-*.md --prefix lec05-kpNN`。
   「before」比對 `###` 標題,中英標題不同:第一輪中文會成功、英文在找不到的標題處報 IndexError;把 caps 裡中文標題字串換成英文標題的開頭(例如「延伸」→「Extension」)再跑第二輪,中文會原位取代,英文補齊。
4. 已不用的舊圖:`grep` 確認講義沒有引用後 `git rm`;被合併的頁,講義裡對應的圖說區塊(`圖說 `...``/`Figure `...``與圖片行)一併刪除。
5. `python3 scripts/check_slides.py`(只看本單元的行,其他單元的既有 FAIL 與本次無關)、`python3 scripts/build_docs.py && mkdocs build`。
6. 更新 `CHANGELOG.md`(最上方一行)、`COURSE_STATE.md`(末尾一行,含「v3 已重做」清單與下一步)、`schedule.md`(只在講義或簡報檔名變動時)、`my-jarvis/logs/2026-10-24-lec05-投影片重做.md`(檔尾「時間為推算」之前加一行)。
7. 逐一 `git add`(`CHANGELOG.md COURSE_STATE.md assets/figures lectures slides .agents/skills/view-from-text-v3/examples`),提交前看 `git diff --cached --name-status HEAD`:只該有 A、M,以及你刻意刪的舊圖 D;提交到 `main` 並推送;同步 `examples/kpNN_*.py` 到 course-template 與 idl-2026-fall 並各自推送。my-jarvis 的日誌另外提交推送。

**一鍵發佈(多個 agent 平行時必用)**:`tools/publish_unit.py` 把上面 1–7 步包成一個指令,並用鎖(`/tmp/pr-publish.lock`)讓多個 agent 依序發佈、不會互相踩到:
```bash
python3 .agents/skills/view-from-text-v3/tools/publish_unit.py --unit-dir <單元資料夾> --kp kp09 --pages 10 \
  --lecture lec05-kp09-cost-functions --map map.json --caps caps.json --remove "lec05-kp09-舊圖.png,..." --en-before "延伸=Extension" \
  --changelog "[更新] KP09 簡報以 view-from-text-v3 重做並發佈:..." --state "KP09 簡報以 v3 重做並發佈(內容頁 N 頁,中英文版),講義換圖;..." --log "依教授指示 ... 重做 KP09 並發佈 ..."
```
先加 `--dry-run` 看圖與檢查結果(之後用 `git checkout -- . && git clean -fdq assets` 還原);沒問題再正式執行。腳本會在 `git status` 不乾淨時中止,不要在 repo 裡留未提交的手動修改。

## 與教授互動的固定節奏
規劃(每頁標題、主圖、對應講義節、數字)→ 教授回「如你建議」→ 製作並稽核到 FAIL 0 → 傳中英文截圖(`tools/sheet.py` 拼圖)→ 教授說「發佈 KPNN」才寫入 repo。教授沒說發佈,不寫入;不要自己決定合併或刪頁。

## 新功能(2026-10-27)
填色透明度、漸層(含多色、徑向)、陰影、多邊形、曲線路徑與開放折線 polyline 已可用(polyline 的工具鏈對應尚未實作,見 `DESIGN.md` 最後一節),頁面也不再保留 KP 標題區(整頁 1280×720);用法見 `DESIGN.md` 最後一節與 `docs/`。

## 規則(摘要)
- 原生節點成品;字級 ≥ 16px;可用填色透明度、單色與多色漸層、徑向漸層、陰影、多邊形與曲線路徑(用法見 `DESIGN.md` 最後一節與 `docs/`);圖寬 1160、高 ≤ 480(整頁畫布為 1280×720)。
- 每頁自足:本頁符號與名詞在圖內定義;term 定義節點用 `named(..., name="term:詞")`。
- 數字由程式算出,替代文字列出所有數字與結論;英文版逐字翻譯並重跑稽核。
- 數學一律 KaTeX(`k(...)`),轉換後成為 `$…$` 數學節點;含 `\left` 的式子留足高度。
- 設計完要看 lint 輸出(字級、出界、溢出、自動換行)與截圖,再看轉換後的 preview。

## 檔案
- `tools/htmlkit.py`、`tools/new_unit.py`、`tools/run_unit.sh`、`tools/gen_figs.py`、`tools/insert_figs.py`
- 依賴:`view-from-text-v2/tools/figkit/html2nfig.py`、`view-from-text/tools/deckkit/nativefig.py`(`NFig.text(..., h=)`)、`slides-to-html/node_modules/katex`、Playwright/Chromium。
