---
name: image-to-jbd-page
description: 把一張圖片(投影片截圖、資訊圖、示意圖)精準復現成 JianBauDau 原生簡報頁(.jbd.json 的 shape / text / polygon / polyline 節點,不是嵌入圖片)。使用時機:使用者給一張圖並說「轉成原生簡報」「做成一頁簡報」「用原生節點重畫」,或要把某頁加入/取代現有 JianBauDau 簡報的某一頁。
---

# image-to-jbd-page:圖片 → JianBauDau 原生頁

目標:輸入一張圖,輸出**可編輯**的 JianBauDau 頁(方框、文字、折線、多邊形、橢圓、KaTeX 公式),版面、顏色、文字與圖片**一致**。不貼圖(JianBauDau 的 image 節點只收 http(s) 網址,且不可編輯),整頁都用原生節點重畫。

本技能自給自足,只需要 Python 3.9+、Pillow、Playwright(+Chromium,用於預覽)。格式規則的完整版在 `docs/AI_AGENT_FILE_FORMAT_SPEC.md`(開頭幾節與「驗證規則」務必讀過)。

## 檔案

| 路徑 | 用途 |
|---|---|
| `tools/jbd_page.py` | 頁面建構庫 `Page`:所有座標用**原圖像素**,等比例完整縮放並置中於 1280×720（非 16:9 留白、不裁切）;`rect / ellipse / text / label / poly / line(含箭頭、虛線、曲線)/ grid`;`dump()` 會檢查字太小、可能溢出、超出頁面 |
| `tools/measure.py` | 量原圖:`grid`(加座標格線)、`crop`(放大局部)、`colors`(取點色碼)、`palette`(主色) |
| `tools/new_deck.py` | 產生空白簡報(只在沒有現成簡報時用) |
| `tools/insert_page.py` | 把頁插入(`--after N`)或取代(`--replace N`)簡報的第 N 頁,自動順延後面各頁與相機步驟 |
| `tools/compare.py` | 渲染某頁,和原圖並排(可加 50% 疊合),找出差異 |
| `tools/preview_deck.py`、`tools/deckkit/` | 近似渲染器(Chromium + KaTeX),`compare.py` 會呼叫 |
| `tests/test_regressions.py` | 回歸測試：等比例置中、重複插頁／取代、其他頁內容與相機位置保留；執行 `python3 -m unittest discover -s tests -v` |
| `examples/example_page.py` | 完整範例:建頁 → 新簡報 → 插入;跑一次就知道每個函式怎麼用 |
| `docs/AI_AGENT_FILE_FORMAT_SPEC.md` | JianBauDau v3 格式說明書(含 polyline、漸層、陰影) |

## 流程(照順序,不要跳)

1. **看懂圖,拆成積木。** 先用一段話寫出版面:有幾個區塊、各區塊的外框顏色、標題、文字內容、箭頭與連線、圖示、公式、底部橫幅。圖示(鎖頭、波形、箭頭)拆成簡單的 rect / ellipse / poly / line;**不要放棄細節,但也不要為了擬真堆上百個節點**(一頁約 40–200 個節點是常態)。
2. **量測。** `measure.py grid src.png --step 50` 取得有座標的圖;看不清的小字、細線用 `crop` 放大。顏色用 `colors` 取點、`palette` 看主色,**不要憑感覺猜色碼**。把量到的座標記下來(原圖像素)。
3. **寫建頁腳本。** 複製 `examples/example_page.py`,`Page(src_w=原圖寬, src_h=原圖高)`,逐區塊用量到的座標畫。
   - 文字:**照抄原文**(含標點、大小寫、單位),不改寫、不翻譯、不補字。`size` 用原圖字高(像素,約等於 em);框寬取原文字塊的寬,讓換行位置和原圖一致。
   - 公式用 `$...$`(KaTeX 語法,見下方陷阱);公式與文字混排時分成不同節點。
   - 單行置中:`label()`;多行:`text(..., line=1.25)`,高度自己算。
   - 線與箭頭:`line(pts, color, sw, arrow="end", dash=True, smooth=True)`;折線用多個點,曲線用 `smooth=True`。
   - 填色透明度用 `rgba(...)`;漸層、陰影用 `rect(..., gradientTo=..., gradientAngle=..., shadowColor=..., shadowBlur=...)` 等 style 欄位(見格式說明書的 fill-effects 一節)。
4. **輸出並插入。** `P.dump("nodes.json")`(會印警告:太小的字、可能溢出、超出頁面,**全部要處理**),然後
   `insert_page.py deck.jbd.json --nodes nodes.json --after N --label "短標籤" --notes "講者備註"`(或 `--replace N`)。沒有現成簡報先 `new_deck.py`。
5. **並排比對並迭代。** `compare.py deck.jbd.json <頁碼> src.png --out cmp.png --overlay`,**看圖**(必須實際看 cmp.png,不要只看數字),逐項修正:位置與大小 → 顏色 → 文字大小與換行 → 細節(線寬、圓角、虛線)。修改後重新建頁:用 `--replace N` 覆蓋同一頁,不要重複插入。至少做兩輪,直到肉眼看不出差異或剩下的差異是已知的渲染器限制。
6. **收尾檢查。** `compare.py` 印出的 `overflowing boxes` 必須為 0;簡報檔能被 JSON 載入且符合格式說明書的驗證規則;向使用者回報:放在第幾頁、有哪些無法完全復現的地方、預覽只是近似渲染(請在 JianBauDau 實際確認)。

## 精準復現的規則

- **座標優先於美觀。** 要的是「像原圖」,不是「改得更好看」。不要自作主張調整間距、換色、換字。
- **每個可見元素都要有對應節點**:外框、底色、標題、圖示、箭頭、連線、虛線、淡色背景線(例如全連接的細線可用少量代表線,並在回報中說明簡化)。
- **文字顏色要對比**:深色頁面上的文字用淺色;若原圖是淺色底,整頁用淺色填滿一塊 `rect` 當背景(JianBauDau 頁面預設背景由簡報主題決定,不保證白色)。
- **字級**:小於 12px(1280×720 頁面上)的字會被警告;原圖本來就很小的註腳可略放大到 13–14。
- **中英文**:原圖是什麼語言就用什麼語言;若專案規定中英文分檔,另外建一份英文版頁面(翻譯由使用者或專案規範決定,不要擅自混排)。
- **只重畫圖上有的內容。** 不補充說明、不加講者備註以外的文字;講者備註只在使用者要求(或專案規範要求)時寫。

## 已知陷阱

- 非 16:9 原圖採 contain 等比例縮放與置中留白；仍以原圖像素提供座標，切勿自行再加置中偏移。比對圖也採相同留白方式。
- 插入／取代頁面的預設節點前綴會自動避開現存 ID；若指定 `--prefix`，請自行確保唯一，衝突會拒絕寫入。

- `Page.text` 的框寬不夠會自動換行、框高不夠會溢出:`dump()` 的警告就是在抓這個,警告出現就加高或加寬,不要忽略。
- **KaTeX**:不支援 `\dfrac`、`\iff`、`cases` 環境(用 `\frac`、`\Leftrightarrow`);`\text{}` 內不要放中文;下標用 `_`,多字元用 `{}`。
- `polyline` 只描邊(fill 透明),最多 200 點、至少 2 點;`polygon` 至少 3 點;曲線點用 `smooth=True`。端點的箭頭是另外一個三角形 polygon(`arrow=` 自動加)。
- 虛線(`dash=True`)在 JianBauDau 有效;預覽器對**矩形**的虛線框可能畫成實線,不影響成品。
- 文字不可放在 polyline / polygon 裡(沒有文字欄位效果),標籤另開 text 節點。
- 不要用 `image` 節點貼原圖;也不要用 `asset:` 參照或 base64。
- 頁面之間間隔 1440 px(頁寬 1280 + 空隙 160),`insert_page.py` 已處理;**不要手動改 x 座標**。
- 簡報若是由腳本產生的(例如課程 repo 的 `examples/kpNN_pages.py`),直接改 `.jbd.json` 會在下次重建時被覆蓋;把建頁腳本(`P.dump` 之前的部分)一併存進專案,並告知使用者。
- 預覽器不是 JianBauDau:字體、行距略有差異;以「沒有溢出、沒有重疊、版面一致」為準。

## 在本課程 repo(my-jarvis / pr-2026-fall)使用時

- 簡報檔在 `slides/lecNN-kpMM-*.jbd.json`(中文)與 `...-en.jbd.json`(英文);兩份都要加頁,英文版用英文文字。
- 加頁後:`python3 scripts/check_slides.py` 不得有 FAIL;`CHANGELOG.md` 記一行(類型「更新」,台北時間);照 `AGENTS.md` 規則直接提交 `main`、訊息用繁體中文、先取得教授同意;建頁腳本存到 `.agents/skills/view-from-text-v3/examples/`。
- 若要走完整的課程簡報流程(規劃頁面、稽核、發布、同步其他 repo),改用 `view-from-text-v3`;本技能是「已有一張設計好的圖,照樣轉成原生頁」的捷徑。
