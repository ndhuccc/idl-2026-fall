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

## 2026-10-27 起:JianBauDau 新功能(填色透明度、漸層、陰影、多邊形)與整頁可用

JianBauDau 取消了每個 Frame 上方給 KP 標題的標題區,並新增四項樣式。格式說明在 `docs/AI_AGENT_FILE_FORMAT_SPEC.md`,範例在 `docs/fill-effects-and-polygons.json`。

**頁面**
- 一頁就是完整的 1280×720(`DECK_HEADROOM` 預設 0,相機 zoom = 1,中心在頁面正中央)。舊簡報(headroom 100)照舊能稽核,`audit_deck.py` 會從相機自動判斷。
- 頁面標題、銜接條仍在上方,圖的可用高度 = 720 − 標題與銜接條(約 233)− 結論條所需。要更高的圖,可把中英文標題縮成一行或把銜接條縮短。
- 螢幕上的字比以前大約 14% 大(zoom 從 0.878 變 1.0),字級下限仍用 16。

**樣式(NodeStyle)**:`fillOpacity`、`gradientTo` + `gradientAngle`(0 由下往上、90 由左往右、180 由上往下)、`shadowColor/Opacity/Blur/OffsetX/OffsetY`、`shapeKind: "polygon"` + `polygonPoints`(3~200 點,座標為外框的 0~100)。

**HTML 設計時怎麼寫,轉換器會對應**
| HTML | 轉成 |
|---|---|
| `background: linear-gradient(135deg, #a, #b 50%, #c)`(可多色)或 `radial-gradient(...)` | `fill` + `gradientTo` + `gradientAngle`,中間色進 `gradientStops`,徑向 `gradientType: "radial"` |
| `box-shadow: 0 16px 28px 0 rgba(0,0,0,.4)` | `shadow*` |
| `rgba(...)` 填色、SVG 的 `fill-opacity` | 保留透明度 |
| SVG `<polygon points>`,可用 `fill="url(#lg)"` | `shapeKind: "polygon"`(`nativefig` 的 `NFig.poly`) |
| SVG `<path>` 用 M/L/H/V/C/Q/Z 畫的曲線,或 `<polygon data-smooth="all">` | 多邊形,曲線取樣成「曲線點」(`smooth: true`,Catmull-Rom);直線頂點維持尖角 |
| SVG `<radialGradient>`、多色 `<linearGradient>` | 徑向、多色漸層 |
| `border-radius: 50%` | 圓形(圓角矩形,半徑 = 短邊一半) |
| `filter: drop-shadow`、圓弧 `A` 路徑 | 仍不支援 |

**Python 端**:`NFig.rect/ellipse(..., fx={"grad": (to, angle), "opacity": 0~1, "shadow": (color, opacity, blur, dx, dy)})`、`NFig.poly(points, fill, stroke, sw, fx)`;`to_png` 會畫多邊形與透明度,漸層取兩色的中間色,陰影略過。`preview_deck.py` 也能顯示多邊形、漸層、透明度與陰影(多邊形的陰影除外)。

**注意**:非矩形形狀裡不要放字(規則不變)。陰影在深色底上要用有色光暈才看得到。

**2026-10-28 更新**:JianBauDau 再加入多色漸層(`gradientStops`)、徑向漸層(`gradientType: "radial"`)與曲線路徑(多邊形頂點 `smooth: true`)。`NFig` 的 `fx` 可帶 `"gtype": "radial"` 與 `"stops": [(offset, colour), ...]`,`poly` 的點可寫 `(x, y, True)` 表示曲線點;`to_png` 與預覽都會畫曲線(`nativefig.smooth_outline`),多色漸層在 PNG 取中間色近似。另外修正:CSS `border-radius: 50%` 現在會變成真正的圓。範例檔與說明書已更新在 `docs/`。
