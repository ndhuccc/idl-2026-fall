# HTML 設計規則(給 html2nfig 轉換用)

轉換器讀的是「瀏覽器排好版後的 DOM 位置」,所以只要設計落在下列限制內,就能轉成原生節點。

## 畫布
- 固定尺寸 1160×470(一頁一圖);`page()` 已加 `*{box-sizing:border-box}`,所有元素絕對定位。
- 深色主題:底 `BG`、字 `TXT`、次要字 `MUT`;角色色用 `ROLE`(red/blue/green/orange/violet/yellow/teal/grey)。

## 可轉換的元素
| HTML 設計 | 轉成 |
|---|---|
| `card()`(圓角、淡色底、細框) | 矩形 |
| `chip()`(有底色、單行文字、無框) | 帶文字的矩形 |
| `txt()` | 文字節點(換行用 `<br>`,不要靠自動換行) |
| `k()` KaTeX(`<span class="k" data-t="LaTeX">`) | `$…$` 數學節點 |
| `named()`(`data-name="term:…"`) | term 定義節點 |
| `svg()` 的 line / rect / circle / ellipse | 線、矩形、橢圓 |

## 一定要避免
- 漸層(轉成中間色)、陰影(丟棄)、polygon/star 等多邊形(改用圓、矩形、線組合)。
- 圓形加外框(改成圓角矩形或另畫橢圓)。
- 多行文字放進 chip(行高等於 chip 高,會溢出)→ 拆成多個文字節點,或改單行。
- 字級 < 16px;內容超出畫布;文字自己換行(轉換後寬度不同,會重疊)。
- 深色 chip 上放 INK 文字會被 E10 判對比不足 → 用 `chip(solid=True)` 或淡色底配 TXT。
- 含 `\left` 的 KaTeX 要給足高度(轉換時至少 字級×2.75)。

## 寫法提醒
- LaTeX 在 HTML 裡用單一反斜線;寫在 Python f-string 內,`{}` 要雙寫(`\mathcal{{K}}`)。
- 迴圈變數不要取名 `k`(會蓋掉 helper);用 `formula = k` 別名。
- 英文版要重新排版檢查(文字較長),`build()` 會一次輸出 zh/en 並 lint。

## 檢查順序
1. `build()` 的 lint 輸出為空。
2. 看 `<fid>_<lang>.png`(瀏覽器版)。
3. `native()` 轉出節點 → `run_unit.sh` 的 preview 截圖與瀏覽器版對照(公式高度、文字寬度、標籤重疊)。
4. audit_deck FAIL = 0。
