---
name: view-from-text-v3
description: Turn a lecture into a JianBauDau deck of native-node information graphics. Plan pages first, design freely in HTML/CSS/SVG with KaTeX, convert to native nodes, audit, get the teacher's confirmation, and publish within the selected course. Data charts can use the v2 layout library.
---

# view-from-text-v3：依目前課程製作簡報

教授指示：規劃 → 以 HTML/CSS/SVG 自由設計資訊圖（數學用 KaTeX）→ 自動轉成原生節點。成品可編輯、可翻譯，講義 PNG 與簡報來自同一份設計。

沿用 v1/v2 的 CLARITY、LEVELS、NARRATIVE、deckv2、audit_deck、preview_deck、plan_pages.py。先讀 DESIGN.md（設計規則與限制）。

## 課程與章節設定

- 預設課程是**本技能所在的課程資料夾**，由程式位置向上尋找 .agents/skills；不以其他課程名稱或固定電腦路徑為預設。
- 可用 --repo <課程根目錄> 明確選課，或設定 COURSE_REPO。指令列 --repo 優先。
- new_unit.py 必須以 --lecture lecNN 指定章節（相容明確設定的 LEC 環境變數）；完整講義檔名由發布時的 --lecture lecNN-kpNN-主題 指定，工具檢查章節與 KP 一致。
- 新單元的 course-unit.json 保存課程根目錄、章節與課名；產生的程式依此載入。課程搬家後可更新設定檔或以 COURSE_REPO 指向新位置。
- 中文課名取自本課 semester.toml，缺少時讀 COURSE_STATE.md 的課程欄；英文課名只使用明確設定的 course_name_en（或 extra.course_name_en），缺少時省略課名前綴。可用 --course-zh／--course-en 覆寫。
- 工具執行時將 COURSE_SKILLS 指向本課 .agents/skills，並相容既有腳本的 SKILLS_DIR。複製舊範例到課程以外的工作目錄時，先設定 COURSE_REPO 或 COURSE_SKILLS。
- examples/ 是歷史教材設計範例；其中的 PR 課名與 lec05 是範例內容，不是新課程的預設。新教材以本課已核准講義為準。

## 流程（每個單元 KP）

1. **盤點與規劃**：從已通過稽核、冷讀及教師確認的講義抽出小節。寫 kpNN_plan.json（title / covers / main / layout / question / takeaway / numbers / source / on_page），跑 v2 的 tools/plan_pages.py；圖內數字用 kpNN_nums.py 重算。規劃與每張圖的圖簡報交教師確認後才畫。
2. **建立工作資料夾**：
   python3 tools/new_unit.py <工作目錄> KPNN "<中文單元名>" "<English unit name>" --lecture lecNN
   產生 course-unit.json、course_context.py、kpNN_deck.py、kpNN_text.py、術語表、deckcommon.py 與 W_zh／W_en。可另加 --repo、--course-zh、--course-en。
3. **每頁選畫圖路線**：
   - HTML 自由設計：概念、架構、比較、流程與公式推導。用 htmlkit.py 的 design(lang)、build()、native()，以 layout="freeform" 放進頁面；見 examples/kp18_pnn.py。
   - v2 版型庫：plot_xy、bars_compare 等資料圖；資料由程式計算，見 v2/LAYOUTS.md。
   - 教師提供的截圖或生成圖：先在 HTML 重建，再轉成原生節點。
   - 視覺探索：必要時用 page-to-teaching-diagram 產生提示詞或設計稿，再以 HTML 實作。
4. **建置與稽核**：
   MIN=<分鐘> bash tools/run_unit.sh <工作目錄> kpNN_deck
   執行 fit → audit → preview → bottoms。FAIL 必須為 0，檢視 W_zh/prev 與 W_en/prev，並核對中英文圖、數字與講義。
5. **教師驗收**：傳中英文截圖（可用 tools/sheet.py 拼圖）；教師明確指示發布該單元後，才寫入本課正式教材。
6. **發布與驗證**：依下節執行，產生講義 PNG、更新雙語圖說與本課紀錄、檢查簡報及網站，再提交同步。

## 發布

**依本課講義結構處理**：有 lectures/en/<講義檔名>.md 時，處理中英分檔及雙語配圖；僅有 lectures/<講義檔名>.md 時，視為中英同檔，預設保留既有核准配圖，不要求新增 lectures/en/。IDL 已核准的 Matplotlib 圖維持原樣。預覽會列出 lecture_layout 與 lecture_figures；可用 --lecture-figures preserve 明確保留配圖。

在已核准的課程工作副本執行：

    python3 tools/publish_unit.py --unit-dir <工作目錄> --kp kpNN --pages <頁數> \
      --lecture lecNN-kpNN-topic --map map.json --caps caps.json \
      --changelog "[更新] 單元簡報與講義配圖" --state "本單元完成；下一步……" \
      --dry-run

--dry-run **只列出計畫**：不改檔、不執行 Git、不建置、不建立發布鎖。核對課程、章節、輸出檔與教師授權後，移除 --dry-run 才實際發布。

實際發布會：
1. 檢查本課工作副本及必要輸入；以依課程根目錄計算的發布鎖避免同課同時寫入。
2. 把 W_zh/draft.jbd.json、W_en/draft.jbd.json 放到本課 slides/<講義檔名>.jbd.json 與 <講義檔名>-en.jbd.json。
3. 講義中英分檔且需重製配圖時，使用本課 tools/make_specs.py、gen_figs.py 產生雙語講義 PNG，並以 insert_figs.py 更新圖說；中英同檔時保留既有配圖。圖說包含讀圖指引、〔功能：N〕與完整 alt 數字。
4. 檢查這兩份簡報，建置網站；失敗就停止，不提交推送。
5. 保存本單元的數字、頁面、文字、圖說及映射範例，更新本課 CHANGELOG.md、COURSE_STATE.md，提交推送 main。
6. 檔名或內容配置有變動時，先依 AGENTS.md 同步 schedule.md 及連結，確保檢查的是最終版本。

**額外輸出必須明確指定**：
- 預設只處理目前課程；無其他課程或範本的預設同步目標。
- 教師另有授權才加 --sync-to <另一課程根目錄>（可重複），同步本單元的範例檔並提交推送。
- 預設不寫外部日誌。需要時加 --log-file logs/<檔名>.md --log "<紀錄>"；相對路徑從目前課程根目錄解析。明確指定的課程外日誌只寫檔，不替該外部資料夾提交推送。
- --remove 接受本單元 PNG 檔名的逗號清單。先確認講義已無引用才刪除；勿使用全資料夾還原或清除指令來取消預覽。

手動製圖亦可使用：

    python3 tools/make_specs.py <工作目錄> kpNN <頁數> --lecture lecNN
    python3 tools/gen_figs.py <工作目錄>/kpNN_specs.py <工作目錄>/map.json assets/figures
    python3 tools/insert_figs.py caps.json lectures/<講義檔名>.md lectures/en/<講義檔名>.md

舊單元沒有 course-unit.json 時，make_specs.py 必須明確指定 --lecture。insert_figs.py 預設從講義檔名推得 lecNN-kpNN，非標準檔名則加 --prefix。圖說可用 before_en 指定英文插入標題；發布工具也支援 --en-before "中文=English" 的對應。

## 環境與保存

首次環境準備：
- npm ci --ignore-scripts --prefix .agents/skills/slides-to-html
- Python 需要 numpy、scipy、matplotlib、pillow、pytest；HTML 轉換需要 Playwright、Chromium 與中文字型。

工作目錄可能是暫存空間。核准發布時保存 kpNN_nums.py、kpNN_pages.py、kpNN_text.py、資料模組、caps.json、map.json 至本課 examples/；不要把未核准草稿視為正式成果。

## 規則與驗收

- 原生節點成品，字級至少 16px；整頁畫布 1280×720。圖的高度依標題與銜接條剩餘空間決定，細節見 DESIGN.md。
- 每頁自足：符號與名詞在同頁定義；術語節點用 term:詞 命名。
- 數字由程式算出，alt 包含數字與結論；英文版重排版、重跑稽核。
- 數學用 KaTeX，轉成數學節點；含 \left 的式子留足高度。
- 同時檢查 HTML lint、瀏覽器截圖與轉換後 preview；0 FAIL 仍須人工確認。
- 可用透明度、漸層、陰影、多邊形與曲線；polyline 的格式與工具鏈限制見 DESIGN.md，不把尚未支援的功能當成可用。
- 固定節奏：規劃 → 教師確認 → 製作及稽核 → 中英文截圖 → 教師指示發布。

## 路徑回歸測試

    python3 -m unittest discover -s .agents/skills/view-from-text-v3/tests -v
    bash -n .agents/skills/view-from-text-v3/tools/run_unit.sh

測試只用臨時課程資料夾與標準函式庫，驗證兩課路徑、章節、中英同檔／分檔、保留核准配圖、工作目錄搬移、單頁圖規格、雙語圖說與 dry-run 無寫入；不執行教材發布。GitHub 的 v3-portability 工作流程使用相同檢查。
