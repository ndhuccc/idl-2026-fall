---
name: view-from-text-v2
description: Turn a lecture (text) into a JianBauDau deck whose pages are information graphics drawn with native nodes -- one figure can replace several pages. Plans the pages first (which figure covers which lecture sections), draws each figure from a layout spec (13 layouts: plot_xy, columns_compare, two_options, before_after, steps_flow, grid_flow, columns_text, bars_compare, tree_diagram, network_layers, quadrant_2x2, concept_card, equation_parts; plus freeform; extensible), builds deck nodes and lecture PNGs from the same spec, and audits. Use instead of view-from-text when the deck should be figure-first and compact.
---

# view-from-text-v2:一頁一圖、圖優先、一次到位

教授指示(2026-10-26):簡報頁面以**一目了然的簡潔圖解**為訴求,**該用文字就用文字、該用圖就用圖**;**一張圖可以抵好幾頁**;在**講義轉簡報時就規劃**,不要先做成很多頁再事後合併。

v2 沿用 `view-from-text`(v1)的規則與工具(CLARITY C1–C23、LEVELS、NARRATIVE、deckkit、audit_deck、preview_deck),**新增**:頁面規劃(`plan_pages.py`)、可擴充的版型庫(`tools/figkit`)、原生節點繪圖(`nativefig`,在 v1 的 deckkit)。v1 保留,方便對照與退回。

## 一次到位的流程

1. **讀講義,列出小節**(`###`)。程度、時長、術語表照 v1 的盤點報告(`../INVENTORY_FORMAT.md`)。
2. **先規劃頁面,不是先寫頁面**:為每個小節問「讀者在這裡要做什麼」(CLARITY C23 的表:看比例與結構→圖;查數值→表;跟推導→算式;學名詞→文字;做題→題目)。**能一張圖講完的小節,就一張圖;相鄰且關係緊密的小節合成一張圖。** 寫成 `plan.json`(範例 `examples/kp23_plan.json`),跑
   `python3 tools/plan_pages.py plan.json lecture.md --modality-out modality.json`
   它檢查:每個小節都被某一頁涵蓋(P1)、非圖頁寫了「讀者要…」的理由(P3)、一頁涵蓋多個小節卻不是版型圖(P4)、頁數超過小節數(P5)、同一小節佔三頁以上(P6)。每個圖頁在 `plan.json` 寫出**內容證據表**(來自 `../page-to-teaching-diagram` 的第 2 步):`question`(一個核心問題)、`takeaway`(一句必須帶走的結論)、`numbers`(圖上數字與重算方式;教材疑似錯誤要列原值與重算值)、`source`(講義小節);`plan_pages.py` P7 檢查。**規劃交給教授確認(含每張圖的圖簡報)後才畫。**
3. **每張圖選版型並寫 spec**(`LAYOUTS.md`):資料用程式算出,字串用 `{"zh","en"}` 成對;沒有合適的版型就**擴充版型**(`layouts/_template.py`)或用 `freeform`(自己用 `parts.py` 組)。
4. **一份 spec,兩種輸出**:`deckhook.place(page, y, layout, spec)` 產生簡報的原生節點;`deckhook.png(layout, spec, path)` 產生講義的 PNG。數字不會不一致。
5. **組裝簡報**:用 v1 的 deckkit(`kit.py`、`fit.py`、`run_kp.sh`);圖頁只有標題、銜接條、圖與講者備註。把 `figkit` 與 spec 檔放進工作資料夾。
6. **稽核**:`audit_deck.py --modality modality.json --modality-report ...`(v1 工具);`preview_deck.py` 看截圖;再用 HTML 匯出抽查。FAIL 要 0。
7. **交付前給教授看截圖**(中英文各一張),確認後才發佈;講義同步換圖(同一份 spec 的 PNG),並更新 CHANGELOG、COURSE_STATE。

## 與 page-to-teaching-diagram 的關係
`../page-to-teaching-diagram` 是「頁面 → AI 生圖提示詞」的技能。v2 採用它的**內容證據表**與**稽核案例**(AUDIT_CHECKLIST X12–X15);AI 生成的點陣圖只能當設計稿,不放進簡報或講義成品(成品一律是原生節點,數字由程式算出)。需要視覺探索時,才由教授挑某一頁用它生圖,v2 再用版型實作。

## 規則(摘要;全文在 v1 的 CLARITY.md)
- 原生節點優先,SVG 其次,matplotlib 最後(C23):曲線與大量散點才用 SVG。
- 字級至少 16 px;亮底配深字(`header_chip`、`chip(dark=True)`),淡底配淺字;只用純色(沒有陰影、漸層);旋轉單位是**度**。
- 每頁自足(C21):本頁的符號與名詞在本頁圖內定義(公式列、名詞小教室)。
- 圖內數字全由程式算出;替代文字列出所有數字與結論;term 定義用名為 `term:<詞>` 的節點(`text(..., name="term:...")`)。
- 圖高不超過 480(英文版銜接條較長,留餘裕);寬 1160。
- 一頁一個主表達方式;又要看比例又要看計算 → 算式標在圖上,放不下才拆相鄰兩頁。

## 驗收樣本
- KP23(決策樹):`examples/kp23_specs.py`(三張版型圖)與 `examples/kp23_plan.json`(頁面規劃)。
- KP12(權重共享):`examples/kp12_specs.py` 與 `examples/kp12_plan.json`;舊版 17 頁 → 8 頁。
- 後六組版型的範例:`examples/layout_gallery.py`。
