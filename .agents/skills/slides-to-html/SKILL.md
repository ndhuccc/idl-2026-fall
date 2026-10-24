---
name: slides-to-html
description: 把 JianBauDau 投影片(.jbd.json,schemaVersion 3)轉成單一檔案、可離線開啟的 HTML 網頁版(縮放畫布鏡頭運動、公式、連線、漫畫圖片、逐頁語音與講者備註都保留)。課程網站建置時會自動對 slides/ 內每份 .jbd.json 執行,讓投影片下載頁只提供 HTML 下載。需要手動轉檔、檢查轉出的 HTML、或排查下載頁沒有出現 HTML 連結時使用。
---

# 簡報轉 HTML(slides-to-html)

學生不必安裝 JianBauDau 播放器,用瀏覽器開啟單一 `.html` 就能看簡報。轉換器是 `scripts/jbd_to_html.mjs`(Node.js),
公式在轉換時就用 KaTeX 排成靜態版面,圖片、音檔、字型全部內嵌,成品**完全不需要網路**。

## 1. 平常不用手動做

課程網站建置(`python scripts/build_docs.py`,每次推送與每天自動執行)會:

1. 找出 `slides/` **第一層**的每份 `*.jbd.json`(子資料夾不處理);
2. 呼叫本技能的轉換器,把 HTML 放進網站暫存區 `.site_src/slides/`(**不提交進 Git**,Repo 不會變大);
3. 投影片下載頁(`slides/README.md`)**只列 HTML**:網站不發佈 `.jbd.json`(來源檔仍在 Repo),也不放 JianBauDau 播放器連結。轉換失敗或缺少 Node.js 的簡報才退回提供 JSON 與播放器連結,避免壞連結。進度表與講義中指向 `slides/…jbd.json` 的連結,建置時會自動改指向同名 HTML。

已經有同名手作 `lecNN.html` 的簡報不會被覆蓋。

**前提**:建置環境要有 Node.js 與本資料夾的套件。`deploy-site` 工作流程已安裝(`actions/setup-node` + `npm ci`)。
在自己的電腦建置時,第一次先執行一次:

```bash
npm ci --ignore-scripts --prefix .agents/skills/slides-to-html
```

沒有 Node.js 或套件時,`build_docs.py` 只會印出 `[警告]` 並略過,下載頁退回列出 JSON(含播放器連結),網站仍可建置。

## 2. 手動轉檔

```bash
node .agents/skills/slides-to-html/scripts/jbd_to_html.mjs --out-dir <輸出資料夾> slides/lec03-kp01-pilot-with-comics.jbd.json [更多檔案...]
```

- 輸出檔名 = 輸入檔名去掉 `.jbd.json`、加上 `.html`。
- 預設某份轉換失敗只印警告並略過;加 `--strict` 則任何失敗都以非 0 結束。
- 一份約等於原 JSON 大小(圖片是主要部分)再加約 1 MB(KaTeX 字型)。

## 3. 轉出的網頁怎麼操作

← → 或空白鍵換頁(也可點畫面或下方按鈕)、`F` 全螢幕、`N` 顯示講者備註與語音逐字稿、`V` 播放這一頁的語音。
有語音的簡報會出現「語音」與「自動」按鈕(自動 = 換頁時自動播放)。網址 `#5` 可直接開到第 5 頁。

## 4. 轉換規則(對照 JianBauDau 原始碼)

| 項目 | 處理方式 |
|---|---|
| 鏡頭 | 每個步驟是畫布上的一個鏡頭(位置、縮放、旋轉);同一畫布內平移縮放有動畫,換畫布用淡入淡出。 |
| 文字、圖形 | 位置、大小、填色、邊框、圓角、字級、對齊、行高照原樣;圖形文字垂直置中、內距 12。 |
| Markdown | 標題、清單、粗斜體、程式碼、引用、表格;單一換行視為換行;不允許原始 HTML,連結只保留 http/https/mailto。 |
| 公式 | `$…$`、`$$…$$`,配對規則與 JianBauDau 相同;轉換時用 KaTeX 排好。 |
| 字元格式 | `{color=#rrggbb}…{/color}`、`{size=N}…{/size}`、`{font=slug}…{/font}`。 |
| 連線 | 16 個等角度掛點、直線/曲線/折點、箭頭方向與大小、虛線、連線標籤(白底卡片、`above`/`on`/`below`)。 |
| 巢狀畫布 | 框(frame)內顯示子畫布的即時預覽(最深 4 層)。 |
| 圖片、語音 | 內嵌 `data:` 資料,各存一份(同一張圖出現多處不會讓檔案變大)。 |
| 其他圖形 | 橢圓、三角形、菱形、五邊形、星形、箭頭用 SVG 畫。 |

**與原播放器的差異**(已知,可接受):

- 不支援 `video`、`embed`、`group` 節點,轉換時印警告並略過。
- 步驟目標若不是 `scene-point`(例如指向某個節點),以預設鏡頭顯示並印警告。
- `asset:` 參照(只存在編輯器內)無法內嵌,圖片與語音會被略過並印警告;用 JianBauDau 的 **Export** 匯出的檔案不會有這種參照。
- 圖形內的文字若超出方框,字級會自動縮小(最小 60%)以免被截掉。
- 開啟頁面時沒有播放器的「Open-Frame 標題列」,第一步(根畫布)顯示框內的子畫布即時預覽。

## 5. 檢查轉出的 HTML

轉換後請抽查,不要只看有沒有檔案:

1. 用瀏覽器開啟,走一遍所有頁;公式、圖片、連線標籤都要顯示。
2. 用無頭瀏覽器批次驗證(`playwright-core` 不列入本技能的相依,用時才安裝):
   ```bash
   npm i --no-save playwright-core --prefix .agents/skills/slides-to-html
   node .agents/skills/slides-to-html/scripts/check_html.mjs <HTML 資料夾> slides [chromium 執行檔路徑]
   ```
   它會逐份確認:沒有頁面錯誤與對外網路請求、所有圖片載入成功、沒有 KaTeX 錯誤,並**逐節點比對實際顯示的字級、粗細、對齊是否與 JSON 設定相同**。有任何問題就以非 0 結束。
3. 圖片載入失敗通常是**來源 JSON 本身損壞**(例如圖資中被混入文字),請回到 JianBauDau 檢查該檔,不要在轉換器裡硬修。
4. 有 `[警告]` 就逐項處理或向老師說明,不要忽略。

## 6. 公開與大小注意

- 課程網站是公開網址:HTML 內含**講者備註與語音逐字稿**,與 `.jbd.json` 相同(見 `slides/FORMATS.md`「發佈前必須確認」)。
- GitHub Pages 站台上限約 1 GB。網站只發佈 HTML,體積約等於 `slides/` 內簡報的大小(圖片是主要部分);簡報多、圖片大時要留意,必要時先壓縮簡報內的圖片。

## 7. 修改轉換器時

- 放進 `style="…"` 屬性的字串(尤其字型名稱)一律用單引號,或用 `escAttr()` 跳脫。雙引號會把屬性在中途截斷,後面的字級、粗細、顏色全部失效。
- 「超出就縮小字級」要以節點自己的字級(px)為基準;用百分比會變成相對於上一層字級,字反而變得更小。

改 `scripts/jbd_to_html.mjs` 後,對 `slides/` 內全部簡報重新轉換並做第 5 節的批次驗證。
三個課程 Repo(`course-template`、`idl-2026-fall`、`pr-2026-fall`)的本技能必須保持相同。
升級 `katex` 或 `marked` 版本時,要同步更新 `package.json` 與 `package-lock.json`,並確認 `katexCss()` 內嵌字型的取代規則仍然有效(轉換器會在不符時直接報錯)。
