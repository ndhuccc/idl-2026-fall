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
- `filter: drop-shadow`、圓弧 `A` 路徑(仍不支援;漸層、陰影、polygon 與曲線路徑已支援,見最後的「新功能」各節)。
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

## 稽核常見錯誤與解法(KP13–KP22 實戰整理)
| 訊息 | 原因 | 解法 |
|---|---|---|
| B10 兩個節點重疊 | 文字節點估計寬度比實際寬(`txt` 的 `w` 不等於節點寬度);或英文換行變多行 | 相鄰文字間距 ≥ 估計寬度;長句用 `<br>` 自己斷行;英文縮短 |
| B10 標題與銜接條重疊 | 英文標題超過兩行 | 英文標題 ≤ 約 100 字元 |
| B3 超出頁寬 | 節點估計寬度超過頁面 | 同上,加 `<br>` 或縮短 |
| E10 對比 1.03 / 4.14 | 稽核拿「文字色 vs 畫布底色」比,不看底下的色塊;深色字放在色塊上會被判 1.03 | 色塊上一律用淺色字(`TXT`),色塊本身用夠深的顏色;畫布上的彩色字用較淺的色(例如 `#7fb0e6`) |
| R4 銜接條沒有接上一頁 | 銜接條沒有與上一頁標題/銜接條/可見文字共用詞 | 銜接條開頭複述上一頁標題裡的詞與數字,再提出本頁要回答的問題(R5:要有疑問句) |
| D3 步驟標籤以數字開頭、過長 | 頁標籤 `lab` | 標籤用名詞片語,不以數字開頭,≤ 20 字元 |
| N3 / X3 數字沒有算式 | 頁面出現數字卻沒有「=」的代入步驟或〔程式計算〕標記 | 頁面上寫標記 `〔程式計算〕`(英文 `(computed)`,不要被換行拆開),或寫出算式 |
| G3 Unicode 上下標 | 圖內用了 ², ₁ 等 | 可接受(WARN);公式一律 KaTeX |
| PNG 圖畫不出公式 | 講義 PNG 用 matplotlib mathtext | 用 `\leq`(不用 `\le`)、`\frac`(不用 `\dfrac`)、`\sqrt{2}`(不用 `\sqrt2`)、不用 `\tfrac`、`bmatrix`、`\iff` |
| 講義 PNG 失敗(`\\iff`、`\\begin{cases}`、`\\mathbf x`) | matplotlib mathtext 不支援 | 用 `\\Leftrightarrow`、分行寫兩條公式、`\\mathbf{x}` |
| 發佈後圖說重複 | 舊講義是「圖片在前、圖說在後」,`insert_figs.py` 只認「圖說在前」 | 發佈前先把該講義的圖說區塊調成圖說在前(中英都要),單獨提交(發佈鎖內)再發佈 |
| 公式框高度 overflow(差幾單位) | `\frac` 的高度估計 | 縮小字級或加大 `formula` 的 `h`;差 < 10 且畫面無問題可接受 |
| 文字 < 16px 的 lint | svg 文字或小標籤 | 全部 ≥ 16 |
| 背景跑超過 2 分鐘 | `run_unit.sh` 約 3 分鐘 | `nohup bash -c '...' &` 背景執行,輪詢 `run.log` 直到出現 `screenshots:`;不要用 `pkill -f`(會殺掉自己) |

**polyline(開放折線／曲線,JianBauDau 新增)**:`shapeKind: "polyline"`,格式與 `polygon` 相同(`polygonPoints` 為外框 0~100 的座標、`smooth: true` 曲線點),差別在**不封閉**(最後一點不接回第一點)、**至少 2 個頂點**、以描邊為主:`fill` 用 `"transparent"`、`stroke` 是線色(不能是 transparent)、`borderWidth` 是線寬(建議 3~8)、`borderStyle: "dashed"|"dotted"` 可做虛線、線端與轉角是圓的;可搭配陰影;Connector 連到 polyline 時掛點在外框上。頂點少於 2 個或含非有限數字會退回矩形。完整格式見 `docs/AI_AGENT_FILE_FORMAT_SPEC.md` 的「`shapeKind: "polyline"`」一節,範例在 `docs/fill-effects-and-polygons.json`(第 5 頁後半的曲線與折線頁)。
- **工具鏈已對應 polyline**(2026-10-28):HTML 裡 SVG 的 `<polyline points=…>`、沒有 `Z` 且 `fill="none"` 的 `<path>`(M/L/H/V/C/Q),會轉成 `shapeKind: "polyline"` 節點:`fill` 透明、`stroke` 是線色、`borderWidth` 是線寬、`stroke-dasharray` 有數字就轉成 `borderStyle: "dashed"`;`<polyline data-smooth="all">` 全部頂點成為曲線點(端點切線由唯一鄰點決定)。沒有 `stroke` 的開放路徑會被略過。`NFig.poly(pts, fill, stroke, sw, fx, closed=False, dash=False)`、講義 PNG(`to_png`)與 `preview_deck.py` 都會畫開放線(線端圓角、虛線);陰影在 PNG 略過。示範見 `examples/polyline_demo.py`(`python3 examples/polyline_demo.py out` 產生 HTML/PNG/JSON)。限制:圓弧 `A` 指令與 `filter: drop-shadow` 仍不支援;一條線一個節點,不要在 polyline 裡放文字。
