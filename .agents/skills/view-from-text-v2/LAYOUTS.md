# LAYOUTS.md — 版型庫(可擴充)

版型 = 一個函式 `build(spec, lang) -> NFig`,用 `@layout("名稱")` 登錄(`tools/figkit/layouts/`)。**版型是起點,不是限制**:找不到合適的,就擴充或用 `freeform`。

## 怎麼挑(依讀者在這一頁要做什麼,不依題材)

| 讀者要做的事 | 版型 | 結構 | KP23 範例 |
|---|---|---|---|
| 並排比較 N 個情形,每個要看例子、代入、結果 | `columns_compare` | 圖例、N 欄(標題、圖示、代入式、大字結果、評語)、公式列、結論 | 三個節點的不純度 |
| 一個起點、兩個選項,比較「正確做法」與「直覺做法」 | `two_options` | 起點、兩個選項(含分堆)、結果列(公式、判定標籤)、結論 | 為什麼要加權 |
| 兩種狀態對比(小/大、有/無),各有示意圖、關鍵數字與差距 | `before_after` | 兩欄(示意、說明、兩個數字、差距)、資訊框、結論、備註 | 樹要多大 |
| 一個程序或因果鏈,N 步依序 | `steps_flow` | N 個方塊以箭頭連接(標題、算例、結果)、結論 | (第二個樣本) |
| 以上都不合 | `freeform` | 用 `parts.py` 的零件自己組 `NFig` | 分區與決策樹、門檻數線 |
| 曲線、大量散點 | SVG(v1 `jbd_figure.py`) | 原生畫不出來 | 熵曲線、剪枝折線 |

## 零件(`parts.py`)
`header_chip`、`panel`、`chip`、`outline_box`、`formula_bar`、`takeaway`、`legend`、`icon`/`icon_rows`(圓、方、三角代表三類)、`SKETCHES`(`tree_shallow`、`tree_deep`、`magnifier_note`)。色彩角色:`red blue green orange violet yellow teal grey`。新增零件直接寫在 `parts.py`,並在這裡加一行。

## 設計約束(所有版型共用)
1. 寬 1160、高 ≤ 480;標題與銜接條在圖外。
2. 字級 ≥ 16;亮底深字、淡底淺字(稽核 E10 檢查對比);純色,無陰影漸層。
3. 所有字串經 `T()`(`{"zh","en"}`);數字由 spec 帶入,spec 由程式算出。
4. 替代文字 `alt` 列出圖上每個數字與結論。
5. 版面常數(欄寬、列高)寫在版型函式裡,不要散落在 spec。

## 擴充版型(三步)
1. 複製 `tools/figkit/layouts/_template.py` 為 `<名稱>.py`,改幾何,用 `@layout("<名稱>", "一句話:服務什麼讀者任務")` 登錄。
2. 在這份表加一行(讀者任務、結構、範例)。
3. 在 `examples/` 放一個 spec 範例(用真實資料),渲染並看 PNG 與簡報截圖,通過稽核後再提交。
重複出現兩次以上的 `freeform` 圖,就該升級成版型。
