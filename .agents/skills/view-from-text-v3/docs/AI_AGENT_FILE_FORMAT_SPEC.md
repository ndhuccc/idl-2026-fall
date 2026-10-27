# JianBauDau 簡報檔案格式說明書（給 AI Agent 讀的版本）

## 這份文件是給誰、做什麼用的

這份文件的讀者是 **AI agent**，不是人類使用者。目的是讓 agent 在讀完一段使用者提供的文本內容之後，能夠**獨立產生一份符合 JianBauDau 格式的 JSON 檔案**，這份 JSON 檔案可以直接被 App 的「Open」功能匯入（也可以被 Player 直接播放），變成一份可導覽、可播放的簡報。

這份文件只講**檔案格式本身**（schema、欄位、型別、合法值、驗證規則）。至於「怎麼把內容設計得好看、好懂、有結構」，請一併參考同目錄下的 [`PRESENTATION_DESIGN_GUIDE.md`](PRESENTATION_DESIGN_GUIDE.md)——那份文件講的是設計原則（一個 Frame = 一個知識單元、巢狀深度、Connector 表達關聯、配色對比、版面多樣性等），兩份文件搭配使用。

兩份完整範例檔案，用 App 的 **Open** 匯入後即可瀏覽與播放：

| 範例 | 示範什麼 |
|---|---|
| [`examples/photosynthesis-knowledge-units.json`](examples/photosynthesis-knowledge-units.json) | **巢狀 Frame 的知識單元**：多個 Scene、Frame 巢狀、Connector、`reveal` 逐項揭示。 |
| [`examples/fill-effects-and-polygons.json`](examples/fill-effects-and-polygons.json) | **單一 Scene 的「頁」**：每頁一個 `scene-point` Step、整頁填滿畫面；填色透明度、線性／多色／徑向漸層、陰影、尖角與曲線多邊形。 |

**產生檔案後，agent 必須自我檢查**本文件「驗證規則與常見錯誤」一節列出的每一條規則，確認輸出的 JSON 不會被 App 的驗證器拒絕。

---

## 核心心智模型：這是什麼樣的資料結構

JianBauDau 不是「一張一張投影片」的線性模型，而是：

1. 有很多個 **Scene（場景）**，每個 Scene 是一塊獨立、無邊界的 2D 畫布，有自己的座標系統。
2. 每個 Scene 裡放很多個 **Node（物件節點）**：文字、圖形、圖片、影片、嵌入內容、群組、連接線，以及一種特殊節點——**Frame（畫框）**。
3. 一個 Frame 節點本身長得像畫布上的一個矩形物件，但它同時**指向另一個獨立的 Scene**（`frame.childSceneId`）。「進入」一個 Frame，就是把攝影機從目前的 Scene 換到那個子 Scene，形成**無限巢狀**的空間（子 Scene 裡還可以再放 Frame，指向孫 Scene，沒有深度限制）。
4. 簡報的「播放順序」是一份全域、扁平的 **Step（鏡頭步驟）陣列**——不管內容散落在多深的巢狀 Scene 裡，Step 陣列用 `order` 欄位決定攝影機依序移動、縮放、進入哪個 Frame、揭示哪些物件。**Step 是唯一決定「投影片先後順序」的東西**，Scene/Node 本身沒有順序概念。

### 兩種常見的版面做法

| 做法 | 適用 | 作法 |
|---|---|---|
| **A. 單一 Scene 的「頁」** | 線性、一頁接一頁的簡報（教材、報告、說明）；內容不需要「放大進去」的空間感 | 只用根 Scene。把每一頁排成同一個 Scene 裡橫向（或縱向）並列的一塊 1280×720 區域，每頁一個 `scene-point` Step，鏡頭剛好框住那一塊。見下方「Step → 整頁填滿畫面」。**最簡單，最不容易出錯，預設選它。** |
| **B. 巢狀 Frame 的知識單元** | 有層級結構的主題：一個總覽、多個單元、單元內還有子單元，要靠「進入／退出」呈現層次 | 根 Scene 放 Frame，每個 Frame 指向一個子 Scene；用 `enter-frame` 進入。見 [`PRESENTATION_DESIGN_GUIDE.md`](PRESENTATION_DESIGN_GUIDE.md)。 |

兩種可以混用（例如每個知識單元的子 Scene 內再用 A 做法排好幾頁）。

**不論哪種做法：進入一個 Frame 之後，畫面上不再有標題列，子 Scene 的內容填滿整個 16:9（或 4:3）畫面**——不用替標題預留上方空間。Frame 的 `text`（標題）只顯示在「還沒進入」的 Frame 上；單元裡需要看得到的標題，請自己放一個 `text` 節點。

---

## 檔案頂層結構（PresentationDocumentV3）

```jsonc
{
  "schemaVersion": 3,
  "rootSceneId": "scene-root",
  "scenes": { "<sceneId>": SceneRecord, ... },
  "nodes": { "<nodeId>": SceneNode, ... },
  "steps": [ CameraStepV3, ... ],
  "metadata": {
    "title": "簡報標題",
    "createdAt": "2026-09-26T00:00:00.000Z",
    "updatedAt": "2026-09-26T00:00:00.000Z",
    "aspectRatio": "16:9"
  }
}
```

- `schemaVersion` 必須是數字 `3`（本文件只講 v3 格式；v2 是舊格式，App 會自動把 v2 遷移成 v3，但 agent 不需要輸出 v2）。
- `rootSceneId`：字串，必須等於 `scenes` 裡某一個 key（通常固定用 `"scene-root"`）。整份文件一定要有一個根 Scene，觀眾一開始看到的就是它。
- `scenes` / `nodes`：以 id 為 key 的物件（不是陣列）。**每個 id 在整份文件裡必須是全域唯一的字串**（Scene id 和 Node id 各自的命名空間也不能互相撞名，建議都用有語意的 kebab-case，例如 `frame-input`、`scene-input`、`input-title`）。
- `steps`：陣列，見下方「Step」一節。
- `metadata.aspectRatio`：只能是 `"16:9"` 或 `"4:3"`，決定 Present／Player／PDF 的畫面比例。**沒有特別理由就用 `"16:9"`**；真的要 `4:3` 一定要顯式填。
- `metadata.createdAt` / `updatedAt`：ISO 8601 時間字串，agent 可以直接填當下時間，格式錯誤不會擋掉檔案匯入（有預設值），但建議填正確格式。

---

## Scene（場景）— SceneRecord

```jsonc
{
  "id": "scene-input",
  "name": "輸入單元",
  "parentFrameId": "frame-input",
  "nodeIds": ["input-title", "input-sunlight-icon", "..."],
  "camera": { "x": 240, "y": 240, "zoom": 0.95, "rotation": 0 }
}
```

| 欄位 | 型別 | 說明 |
|---|---|---|
| `id` | string | 必須跟這個 Scene 在 `scenes` 物件裡的 key 完全一致。 |
| `name` | string | 給人看的名稱，隨意填，不影響播放。 |
| `parentFrameId` | string \| null | 如果這個 Scene 是某個 Frame 節點的子場景，填那個 Frame 節點的 `id`；根場景（`rootSceneId` 指到的那個）**必須是 `null`**。 |
| `nodeIds` | string[] | 這個 Scene 底下、直屬於它的 Node id 清單。**這個清單必須跟 `nodes` 裡實際 `sceneId === 這個 scene.id` 的節點完全對應**（見下方驗證規則）——加了節點記得同時把它的 id 塞進對應 Scene 的 `nodeIds`，兩邊要一致。 |
| `camera` | Camera | 這個 Scene 被獨立檢視時的預設鏡頭（例如剛進入這個 Frame 時的畫面）。`zoom` 必須 > 0。 |

**Frame 與子 Scene 是一一對應、雙向連結的一組物件：**
- Scene A 裡有一個 `type: "frame"` 的節點 F，`F.frame.childSceneId = "scene-B"`。
- `scenes["scene-B"].parentFrameId` 必須等於 `F.id`。
- 這兩件事任一邊漏掉、對不上，App 的驗證器都會回報錯誤（`frame child scene missing` / `child scene parent mismatch`）。**每建立一個 Frame 節點，就要同時建立它指向的那個 Scene，兩邊互相對好 id。**

---

## 座標系統（每個 Scene 各自獨立）

- 每個 Scene 都有自己的、**無邊界、任意單位**的 2D 座標系統，跟其他 Scene 完全無關（子 Scene 不是「父 Scene 座標的一部分」，而是全新的一塊畫布）。
- 座標單位沒有強制的物理意義，習慣上可以把 1 單位想成大約 1px。**建議把「一頁」想成一塊 1280×720（16:9）或 1280×960（4:3）的區域**：把這一頁的所有物件排在這塊區域內、留出 40～80 的邊距，再用 `scene-point` 加 `referenceViewport` 把鏡頭精準對上去（見 Step 一節）。
- `y` 軸方向：正值向下（跟一般螢幕座標系一致，不是數學座標系）。
- 一個 Scene 裡的物件座標**沒有畫布邊界限制**，可以是負數，可以超出「看得到的範圍」——因為攝影機（Camera / Step）才決定最後畫面看到哪一塊區域，物件本身怎麼擺都合法。多頁並列時，頁與頁之間留 100 以上的空隙（例如每頁 x 位移 `1280 + 140`），避免鄰頁在畫面邊緣露出來。

---

## Node（物件節點）— SceneNode

所有節點共用的欄位：

```jsonc
{
  "id": "input-title",
  "type": "text",
  "name": "輸入單元標題",
  "sceneId": "scene-input",
  "transform": { "x": 40, "y": 30, "scaleX": 1, "scaleY": 1, "rotation": 0 },
  "width": 420,
  "height": 80,
  "zIndex": 2,
  "locked": false,
  "style": { "...": "見下方 NodeStyle" },
  "text": "顯示的文字內容（見下方文字內容格式）",
  "initiallyHidden": false
}
```

| 欄位 | 型別 | 說明 |
|---|---|---|
| `id` | string | 全域唯一。 |
| `type` | `"text" \| "shape" \| "image" \| "video" \| "embed" \| "frame" \| "group" \| "connector"` | 決定這個節點是什麼、要不要填對應的內容欄位（`image` / `video` / `embed` / `frame` / `connector`，見下方各節）。 |
| `name` | string | 給人看的名稱，不影響播放。 |
| `sceneId` | string | **必須**等於這個節點所屬 Scene 的 `id`，並且這個 id 要出現在該 Scene 的 `nodeIds` 裡（雙向一致）。 |
| `transform.x` / `transform.y` | number | 節點左上角在該 Scene 座標系裡的位置（不是中心點）。 |
| `transform.scaleX` / `transform.scaleY` | number | 額外縮放倍率，預設 `1`，不能是 `0`。一般情況下直接固定填 `1`，用 `width`/`height` 決定實際尺寸即可，不需要用縮放。 |
| `transform.rotation` | number | 旋轉角度，**單位是度**（例如轉 90 度填 `90`）。沒有旋轉需求就填 `0`。 |
| `width` / `height` | number | 節點的寬高，**必須 > 0**。 |
| `zIndex` | number | 疊放順序，數字越大越上層。同一個 Scene 裡的節點建議給遞增的整數（0, 1, 2, ...），數字本身沒有絕對意義，只用來比大小。**頁面背景放最低、文字放最高。** |
| `locked` | boolean | 是否鎖定禁止編輯，agent 產生內容時填 `false` 即可。 |
| `style` | NodeStyle | 見下方。 |
| `text` | string | 這個節點顯示的文字（`text`／`shape` 節點會實際渲染文字內容；`frame` 節點的 `text` 是它的標題；其他型別節點這個欄位可以填一個辨識用的名稱字串，不會顯示出來，但仍須是字串，不能省略）。 |
| `initiallyHidden` | boolean（可省略） | 若為 `true`，這個節點在該 Scene 一開始不顯示，要靠某個 Step 的 `reveal.enter` 把它揭示出來才會出現（做「逐項揭示」效果時使用，見下方 Step 一節）。 |

### NodeStyle（`style` 欄位）

```jsonc
{
  "fill": "#6a9bcc",
  "stroke": "#a9c5e3",
  "textColor": "#171918",
  "fontSize": 17,
  "radius": 16,
  "fontFamily": "Avenir Next",
  "fontWeight": 400,
  "fontStyle": "normal",
  "textAlign": "left",
  "lineHeight": 1.13,
  "letterSpacing": 0,
  "borderStyle": "solid",
  "borderWidth": 2,

  "fillOpacity": 0.6,
  "gradientTo": "#3b82f6",
  "gradientAngle": 135,
  "gradientType": "linear",
  "gradientStops": [{ "offset": 0.5, "color": "#f43f5e" }],
  "shadowColor": "#000000",
  "shadowOpacity": 0.35,
  "shadowBlur": 20,
  "shadowOffsetX": 0,
  "shadowOffsetY": 12
}
```

基本欄位：

| 欄位 | 型別 / 範圍 | 說明 |
|---|---|---|
| `fill` | CSS 顏色字串（含 `transparent`） | 背景／填色。`text` 節點通常用 `"transparent"`。有漸層時是漸層的**起點色**。 |
| `stroke` | CSS 顏色字串 | 邊框顏色。 |
| `textColor` | CSS 顏色字串 | 文字顏色。**務必實際核對跟 `fill` 的對比度**（見下方「配色與可讀性」）。 |
| `fontSize` | number | px。 |
| `radius` | number ≥ 0 | 圓角半徑（px）。`frame` 節點常用較大值（如 20）。 |
| `fontFamily` | string（可省略） | 預設 `"Avenir Next"`，沒有特殊需求不用改。 |
| `fontWeight` | number，會被夾在 300～800 之間 | 400 = 正常，700 = 粗體。`**text**` markdown 語法也能達到粗體效果（見下方）。 |
| `fontStyle` | `"normal" \| "italic"` | |
| `textAlign` | `"left" \| "center" \| "right"` | `shape` 節點預設置中，`text` 節點預設靠左。 |
| `lineHeight` | number，夾在 0.8～3 | |
| `letterSpacing` | number，夾在 -2～12 | px。 |
| `borderStyle` | `"solid" \| "dashed" \| "dotted" \| "double"`（可省略） | |
| `borderWidth` | number ≥ 0（可省略） | px。**沒有邊框就填 `0`，並且 `stroke` 用 `"transparent"`。** |

填色效果欄位（透明度、漸層、陰影）見下一節。

**⚠️ Connector 的 `textColor` 是唯一的例外**：Connector 標籤永遠畫在一張接近純白（`rgba(255,255,255,0.94)`）的卡片上，跟畫布深色背景無關，所以 Connector 節點的 `style.textColor` **必須用深色**（例如 `#1c1712`），跟其他節點「深色背景配淺色文字」的慣例正好相反。這是最容易被忽略、也最容易做出「看不見的文字」的地方。

---

## 填色效果：透明度、漸層、陰影

這些都是 `style` 裡**選填**的欄位，**主要用在 `text` 與 `shape` 節點**（編輯器的 Inspector／右鍵選單只對這兩種提供設定介面）。全部都省略 = 一般的純色填色，行為跟以前完全一樣。

### 填色透明度

| 欄位 | 說明 |
|---|---|
| `fillOpacity` | 0～1，**乘進填色**（有漸層時乘進每一個顏色）。省略 = 1。**只影響填色**，邊框與文字不透明度不受影響。 |

`fill` 本身也可以是 `rgba(...)`，兩者會相乘。做「毛玻璃」卡片就用深色 `fill` 配 `fillOpacity` 約 0.5～0.7，壓在別的圖形上。

### 漸層

有 `gradientTo` 就代表填色是漸層（沒有 `gradientTo`，其他漸層欄位全部不生效）。

| 欄位 | 說明 |
|---|---|
| `gradientTo` | 漸層的**終點色**（位置 100%）。起點色就是 `fill`（位置 0%）。建議用 `#rrggbb`（編輯器的色彩選擇器只吃這種格式）。 |
| `gradientType` | `"linear"`（預設）或 `"radial"`。 |
| `gradientAngle` | **線性漸層**的方向，單位度，CSS 慣例：`0` = 由下往上、`90` = 由左往右、`180` = 由上往下（預設）、`135` = 由左上往右下。徑向漸層不使用。 |
| `gradientStops` | **多色漸層**：`fill`（0）與 `gradientTo`（1）之間的**中間色**，`{ "offset": 0～1, "color": "…" }` 陣列。`offset` 必須落在 0 與 1 之間（不含兩端，超出會被夾進 0.01～0.99），最多 8 個，載入時會依 `offset` 排序。 |

- **線性**：`fill` 在起點、`gradientTo` 在終點，沿 `gradientAngle` 方向漸變。
- **徑向**（`"radial"`）：`fill` 在**物件中心**，`gradientTo` 在**邊緣**（橢圓、貼合物件外框；矩形的四個角會維持邊緣色）。

範例——三色線性、徑向、半透明：

```jsonc
// 紫 → 洋紅（50%）→ 青藍，135°
"style": { "fill": "#6d28d9", "gradientTo": "#0e7490", "gradientAngle": 135,
           "gradientStops": [{ "offset": 0.5, "color": "#be185d" }], "radius": 32, "...": "..." }

// 中心淡黃 → 邊緣紅
"style": { "fill": "#fde68a", "gradientTo": "#ef4444", "gradientType": "radial", "...": "..." }

// 半透明的白（可以看到後面的東西）
"style": { "fill": "#ffffff", "fillOpacity": 0.25, "...": "..." }
```

### 陰影

有 `shadowColor` 就代表**開啟陰影**。陰影是 `drop-shadow`，**沿著物件的輪廓**描出，所以橢圓、星形、多邊形、甚至文字都適用（文字節點填色透明時，陰影會跟著字形走）。

| 欄位 | 說明 |
|---|---|
| `shadowColor` | 陰影顏色。有這個欄位才有陰影。 |
| `shadowOpacity` | 0～1，乘進 `shadowColor`。省略 = 1。 |
| `shadowBlur` | 模糊半徑（px，≥ 0）。省略 = 8。**硬邊陰影填 `0`。** |
| `shadowOffsetX` / `shadowOffsetY` | 位移（px）。省略時 X = 0、Y = 4。 |

- **畫布預設是深色**：黑色陰影在深色背景上幾乎看不出來。想要明顯的陰影，請搭配**淺色頁面底**，或用有色的「光暈」（`shadowColor` 用亮色、`shadowBlur` 30 以上、位移 0）。
- 常見配方：柔和浮起 `blur 24～30 · offset (0, 14～18) · opacity 0.3～0.4`；硬邊位移 `blur 0 · offset (12～14, 12～14) · opacity 1`；光暈 `blur 34 · offset (0, 0) · opacity 0.9`。
- 陰影在編輯、Present、Player、PDF 匯出中的呈現一致。

---

## 各種節點型別要另外填的內容欄位

### `type: "text"` / `type: "shape"`

- 純文字或幾何圖形卡片。`text` 欄位放要顯示的內容（支援 Markdown，見下一節）。
- `shape` 節點另外需要 `shapeKind`：`"rectangle" | "ellipse" | "triangle" | "diamond" | "pentagon" | "star" | "arrow" | "polygon"`。
- **非矩形形狀（`ellipse`／`triangle`／`diamond`／`pentagon`／`star`／`arrow`／`polygon`）裡不要放長文字**——文字是置中的矩形範圍，塞進尖角形狀會被截斷。用這些形狀當語意化圖示（例如 `star` 代表能量、`ellipse` 代表液體、`diamond` 代表分子、`arrow` 代表流動），文字另外用一個獨立的 `text` 節點放在旁邊。

### `shapeKind: "polygon"`：任意多邊形與曲線路徑

除了內建的幾種形狀，可以自己決定每一個頂點。`shapeKind` 填 `"polygon"`，並加上 `polygonPoints`：

```jsonc
{
  "id": "gem",
  "type": "shape",
  "shapeKind": "polygon",
  "polygonPoints": [
    { "x": 50, "y": 0 },
    { "x": 100, "y": 100 },
    { "x": 0, "y": 100 }
  ],
  "transform": { "x": 600, "y": 200, "scaleX": 1, "scaleY": 1, "rotation": 0 },
  "width": 240,
  "height": 220,
  "...": "其餘欄位同一般 shape"
}
```

- `polygonPoints`：`{ x, y }[]`，**3～200 個頂點**，依序連線並**自動封閉**。座標是**該節點自己外框的 0～100**：`x` 由左到右、`y` 由上到下（不是 Scene 座標）。所以改 `width`／`height` 時，多邊形會跟著縮放。
- **外框要貼齊頂點**：`transform.x/y` 與 `width/height` 請設成頂點範圍的外接矩形（頂點最好有碰到 0 與 100），選取框才會貼合圖形。agent 的作法：先在 Scene 座標寫好每個頂點 → 取最小／最大 x、y 當外框 → 每個頂點換算成 `((x − 外框x) / width) × 100`。
- 頂點少於 3 個、含非有限數字、或超過 200 個時，匯入會**退回成一般矩形**（`shapeKind` 與 `polygonPoints` 被丟掉），不會整份被拒絕。
- 多邊形可以套用所有填色效果（透明度、漸層、陰影）與邊框設定。

**曲線路徑**：頂點加上 `"smooth": true` 就是**曲線點**，輪廓會平滑地穿過它（用 Catmull-Rom 曲線，頂點的切線方向由前後兩個鄰點決定）；沒有 `smooth` 的頂點仍是**尖角**。尖角與曲線點可以混用——例如愛心下方的尖端是尖角、其餘是曲線點；一片葉子兩端是尖角、兩側是曲線點。全部頂點都 `smooth` 就是一個圓潤的塊面。

```jsonc
// 愛心：下方尖端是尖角，其餘為曲線點
"polygonPoints": [
  { "x": 50, "y": 100 },
  { "x": 6,  "y": 56, "smooth": true },
  { "x": 16, "y": 16, "smooth": true },
  { "x": 50, "y": 30, "smooth": true },
  { "x": 84, "y": 16, "smooth": true },
  { "x": 94, "y": 56, "smooth": true }
]
```

曲線點越少、距離越遠，彎曲越大；想要更貼近形狀就多放幾個曲線點。曲線可能略微超出頂點圍成的範圍，邊緣留一點空間即可。

### `type: "image"`

```jsonc
"image": { "src": "https://example.com/photo.jpg", "alt": "描述文字", "fit": "contain" }
```
- `src` **必須是一般的 `http://` 或 `https://` 網址**（agent 沒辦法上傳二進位檔案到伺服器，不要使用 `asset:` 開頭的內部參照，也不要塞 base64 data URI——雖然技術上不會被擋，但檔案會變得非常大）。缺少有效圖片網址時，寧可用 `shape` 或 `text` 節點代替，不要留一個會顯示破圖的 `image` 節點（驗證器會擋掉沒有合法 `src` 的 `image` 節點）。
- `fit`：`"contain" | "cover" | "fill"`，預設 `"contain"`。

### `type: "video"`

```jsonc
"video": { "src": "https://example.com/clip.mp4", "alt": "描述文字", "fit": "contain", "controls": true, "muted": false, "loop": false }
```
- `src` 規則同 `image`。

### `type: "embed"`（嵌入 YouTube / Vimeo / 其他網頁）

```jsonc
"embed": { "provider": "youtube", "url": "https://www.youtube.com/watch?v=XXXXXXXXXXX", "title": "影片標題", "allowFullscreen": true }
```
- `provider`：`"youtube" | "vimeo" | "custom"`。**只要給一般的 YouTube 分享連結或 Vimeo 連結，`provider` 填對應值即可**（`youtube.com/watch?v=`、`youtu.be/`、`vimeo.com/數字ID` 都可以）。其他任意網頁用 `provider: "custom"`，`url` 直接填該網頁的 `https://` 網址。
- 只接受 `https://` 開頭的網址。匯出 PDF 時，嵌入的網頁會變成一個寫著網址、可點擊的方塊。

### `type: "frame"`（畫框，指向子 Scene）

```jsonc
"frame": {
  "childSceneId": "scene-input",
  "previewMode": "live",
  "padding": 18,
  "clip": true
}
```
- `childSceneId`：**必須**對應到 `scenes` 裡實際存在、且 `parentFrameId` 指回這個 Frame 節點 id 的一個 Scene（見上方「Scene」一節的雙向對應規則）。
- `previewMode`：`"empty" | "snapshot" | "live"`。一般建議用 `"live"`（畫框裡即時顯示子場景內容的縮圖預覽）。
- `padding`：子場景內容跟畫框邊緣的留白（px），預設 18。
- `clip`：是否裁切超出畫框的內容，一般填 `true`。
- **Frame 節點的 `text` 欄位就是這個畫框的標題**，顯示在**還沒進入**的 Frame 上，通常填這個知識單元的標題（例如 `"INPUT"`、`"光合作用過程"`）。標題支援多行（用 `\n`）；多行標題會自動縮小字級、每行過長時以「…」截斷，所以標題請盡量簡短。
- 選填的標題文字樣式：`titleFontSize`（10～120）、`titleColor`、`titleFontFamily`、`titleFontWeight`（300～800）、`titleFontStyle`（`"normal"|"italic"`）。不填時，標題字級會依畫框大小自動縮放。
- **進入 Frame 之後不會有標題列**：子 Scene 的內容填滿整個畫面。要在單元內顯示標題，請在子 Scene 裡放一個 `text` 節點。

### `type: "group"`（群組，把多個節點視覺上綁在一起）

```jsonc
"children": ["node-id-1", "node-id-2"]
```
- `children` 是被群組的節點 id 陣列。agent 產生內容時**不建議主動使用 group**（群組主要是給人在編輯器裡手動操作用的），直接把節點各自放在 Scene 裡即可，除非有明確理由需要把某幾個節點綁成一個可整體移動的單位。

### `type: "connector"`（連接線，表示兩個節點之間的關聯）

```jsonc
"connector": {
  "sourceNodeId": "frame-input",
  "sourceHook": 4,
  "targetNodeId": "frame-process",
  "targetHook": 12,
  "curve": true,
  "direction": "forward",
  "arrowSize": "medium",
  "label": { "text": "提供原料", "t": 0.5, "offset": "above" }
}
```
- `sourceNodeId` / `targetNodeId`：**必須是同一個 Scene 裡的其他節點 id**（不能是 Connector 自己，也不能跨 Scene，也不能連到另一條 Connector）。
- `sourceHook` / `targetHook`：整數 `0`～`15`，代表節點四周 16 個等角度掛點（`0` = 正上方，順時針排列，`4` ≈ 正右方，`8` ≈ 正下方，`12` ≈ 正左方，其餘依此類推）。**Connector 節點本身也要放進該 Scene 的 `nodeIds`**，並給它自己的 `transform`/`width`/`height`（隨意填、不影響連線視覺，例如 `x:0, y:0, width:10, height:10`）。
- `curve`：是否用曲線而非直線連接，可省略。
- `direction`：`"none" | "forward" | "backward" | "both"`，決定箭頭方向，可省略。
- `arrowSize`：`"small" | "medium" | "large"`，可省略。
- `waypoints`（可省略）：`{x,y}[]`，手動指定連接線經過的中繼點。
- `label`（可省略）：`{ text, t, offset }`——`t` 是 0～1，代表標籤沿連線的相對位置（0.5 = 中點）；`offset` 是 `"on" | "above" | "below"`。**`label.text` 的 `style.textColor` 一律要用深色**（見上方 NodeStyle 一節的說明）。

---

## 文字內容格式（`text` 欄位支援的語法）

`text` / `shape` 節點的 `text` 欄位、Connector 的 `label.text`，都支援標準 **Markdown** 語法（用 `markdown-to-jsx` 渲染，關閉了原始 HTML）：

- `**粗體**`、`*斜體*`、`***粗斜體***`
- 換行：字串裡直接放 `\n`
- 標題（`# ` `## ` ...）、清單（`- ` `1. `）、引言（`> `）等區塊語法也會被解析成對應的排版——但**在小尺寸的卡片節點裡，建議只用粗體/斜體/換行做簡單強調，不要用完整的標題/清單巢狀結構**，避免排版跑掉。
- 支援 **LaTeX 數學公式**：行內用 `$公式$`，區塊用單獨一行的 `$$` 包住公式（例如 `$$\nE = mc^2\n$$`），會用 KaTeX 渲染。

字面上要顯示 `*`、`#`、`` ` `` 等 Markdown 特殊字元時，前面加反斜線跳脫，例如 `\*` 顯示成 `*`。

---

## Step（鏡頭步驟）— CameraStepV3

`steps` 是一個**全域扁平陣列**，決定簡報播放（Present／Preview／Player）時攝影機依序怎麼移動。

```jsonc
{
  "id": "step-open-input",
  "order": 0,
  "label": "進入輸入單元",
  "target": { "kind": "enter-frame", "sceneId": "scene-root", "frameId": "frame-input" },
  "durationMs": 900,
  "holdMs": 420,
  "easing": "smooth",
  "reveal": { "timing": "after-camera", "enter": [ { "nodeId": "input-title", "effect": "fade", "delayMs": 0, "durationMs": 400 } ], "exitNodeIds": [] },
  "speakerNotes": "給講者看的備註，觀眾看不到"
}
```

| 欄位 | 型別 / 範圍 | 說明 |
|---|---|---|
| `id` | string，非空 | 全域唯一。 |
| `order` | number | **決定播放順序**，建議直接用陣列索引 `0, 1, 2, ...`（陣列本身的排列順序不重要，真正生效的是這個數字，重複或跳號雖然不會擋掉匯入，但會讓順序難以預期，務必保持與陣列順序一致的遞增整數）。 |
| `label` | string，≤ 200 字 | 顯示在時間軸卡片上的名稱。 |
| `target` | CameraTarget | 見下方四種類型。 |
| `durationMs` | number，180～10000 | 攝影機移動到這一步的動畫時間（毫秒）。一般內容用 800～1200。 |
| `holdMs` | number，0～30000 | 到達後停留多久才能進到下一步（手動點「下一步」時這個值只是可以提前跳過的下限）。一般用 400～800。 |
| `easing` | `"linear" \| "easeInOut" \| "smooth"` | 動畫緩動函數，一般用 `"smooth"`。 |
| `fallbackCamera`（可省略） | Camera | `target` 解析失敗時的備援鏡頭，agent 產生內容時通常不需要填。 |
| `reveal`（可省略） | StepReveal | 逐項揭示效果，見下方。 |
| `speakerNotes`（可省略） | string，≤ 20000 字 | 純講者備註，不會顯示給觀眾。 |
| `narration`（可省略） | `{ "src": string, "label"?: string, "transcript"?: string }` | 這一頁的教學語音（mp3 等音檔）。`src` 可以是 `data:audio/mpeg;base64,…`（內嵌）、`https://` 網址，或編輯器內部的 `asset:<id>`；`label` 是原始檔名（例如 `lec04-kp00-p001.mp3`），只用來顯示；`transcript` 是語音的逐字稿，Present 的 Notes 面板會顯示給觀眾。沒有這個欄位的步驟就是沒有語音，不影響播放。Present 下方的 **Voice** 鈕播放它，**Auto** 開關可在換頁時自動播放。Player 讀不到 `asset:` 參照，匯出（Export）時會自動內嵌。 |

### `target`（CameraTarget）四種類型

1. **`scene-overview`** — 讓攝影機縮放到「剛好 framing 這整個 Scene 目前所有內容」的範圍。
   ```jsonc
   { "kind": "scene-overview", "sceneId": "scene-root" }
   ```
   常用在一個 Scene 的第一步（先給觀眾看整體地圖），或退回上一層時。**單 Scene 多頁的做法不要用它**（它會把所有頁一起框進來）。

2. **`focus-node`** — 攝影機縮放到剛好框住某一個節點。
   ```jsonc
   { "kind": "focus-node", "sceneId": "scene-input", "nodeId": "input-title" }
   ```
   `nodeId` 必須是同一個 `sceneId` 裡實際存在的節點。

3. **`enter-frame`** — 把攝影機從目前的 Scene **換到**某個 Frame 節點指向的子 Scene。這是唯一能夠「換場景」的 Step 類型。
   ```jsonc
   { "kind": "enter-frame", "sceneId": "scene-root", "frameId": "frame-input" }
   ```
   `frameId` 必須是 `sceneId` 這個 Scene 裡、`type === "frame"` 的節點 id。

4. **`scene-point`** — 攝影機移到 Scene 座標系裡指定的一個點、指定的縮放倍率。**精確控制畫面裡看到什麼，就用這一種**（前三種都是「自動算出來」的框景）。
   ```jsonc
   { "kind": "scene-point", "sceneId": "scene-root", "point": { "x": 640, "y": 360 }, "zoom": 1,
     "referenceViewport": { "width": 1280, "height": 720 } }
   ```
   - `point`：這個 Scene 座標系裡，畫面**中心點**的座標。
   - `zoom`：必須 > 0，數字越大放得越大（越靠近）。
   - `referenceViewport`（選填，**建議一律填**）：`{ width, height }`，這個 `zoom` 是「相對於多大的參考畫面」。有了它，**畫面實際看到的範圍就與螢幕大小無關**：

     > 看到的 Scene 範圍 = (`referenceViewport.width` ÷ `zoom`) × (`referenceViewport.height` ÷ `zoom`)，中心在 `point`。

     沒有 `referenceViewport` 時，`zoom` 是相對「播放當下的畫面大小」，同一份檔案在不同螢幕上看到的範圍不一樣，agent 無法精算。**所以做頁面時請填成與 `metadata.aspectRatio` 相同比例的參考尺寸**：`"16:9"` 用 `{ "width": 1280, "height": 720 }`、`"4:3"` 用 `{ "width": 1280, "height": 960 }`。`referenceViewport` 的比例要與簡報比例一致，否則上下或左右會多出／少掉一塊。

### 整頁填滿畫面：用 `scene-point` 做「頁」

單一 Scene 的多頁做法（上方「A」）：

1. 第 *i* 頁（*i* 從 0 開始）占 Scene 裡的一塊區域，左上角 `(i × (1280 + 140), 0)`，大小 1280×720（`"16:9"`）。頁間留 140 的空隙。這一頁的物件座標都加上 `i × 1420` 的 x 位移。
2. 先放一個與整頁同大小的**頁面背景**（`shape`，`rectangle`，`radius: 0`，`stroke: "transparent"`，`borderWidth: 0`，`zIndex` 最低），可以用漸層；再放文字與圖形。
3. 第 *i* 頁的 Step：

   ```jsonc
   { "id": "step-3", "order": 2, "label": "② 漸層填色",
     "target": { "kind": "scene-point", "sceneId": "scene-root",
                 "point": { "x": 2 * 1420 + 640, "y": 360 }, "zoom": 1,
                 "referenceViewport": { "width": 1280, "height": 720 } },
     "durationMs": 1000, "holdMs": 600, "easing": "smooth" }
   ```

   `point` 就是這一頁的中心：`(i × 1420 + 640, 360)`；`zoom: 1` 加上 `referenceViewport` 1280×720 → 鏡頭剛好框住這一頁，**內容填滿整個畫面**，不需要為 Frame 標題預留空間。
4. 頁內的逐項揭示照常用：`reveal` 的節點與這個 Step 的 `target.sceneId` 同為根 Scene，所以可以直接列出該頁的節點（節點設 `initiallyHidden: true`）。

**語音檔的頁碼規則（批次匯入用）**：編輯器「Narration → Import files…」依檔名最後的 `-pNNN`（例如 `lec04-kp00-p001.mp3` → 第 1 頁）把音檔對到「內容頁」。內容頁是依播放順序（深度優先）排的、**不含**聚焦 Frame 的步驟（`enter-frame`、以 Frame 為中心的 `scene-point`、`focus-node` 一個 Frame，或在含 Frame 的 Scene 上的 `scene-overview`）。單 Scene 多頁的做法每個 Step 都是內容頁。

### `reveal`（逐項揭示，可省略）

讓一個 Step 到達目的地後，**依序淡入/縮放顯示**目的地 Scene 裡原本隱藏的節點，或讓某些節點退場。

```jsonc
{
  "timing": "after-camera",
  "enter": [
    { "nodeId": "point-1", "effect": "fade", "delayMs": 0, "durationMs": 400 },
    { "nodeId": "point-2", "effect": "fade", "delayMs": 300, "durationMs": 400 }
  ],
  "exitNodeIds": []
}
```

| 欄位 | 說明 |
|---|---|
| `timing` | `"after-camera"`（鏡頭到位後才開始揭示，較常用）或 `"with-camera"`（鏡頭移動的同時就開始揭示）。 |
| `enter[].nodeId` | 要揭示的節點 id。**這個節點必須跟這個 Step 的 `target.sceneId` 屬於同一個 Scene**——不能在 `enter-frame`（換場景）的 Step 上直接列出子 Scene 裡的節點，要另外用一個 `target.sceneId` 是該子 Scene 的 Step（例如緊接著的 `scene-overview`）來搭配 `reveal`。 |
| `enter[].effect` | `"none" \| "fade" \| "scale"`。 |
| `enter[].delayMs` | 0～10000，這一項相對這個 Step 揭示階段開始的延遲，可以用不同的 `delayMs` 做出「一個一個跳出來」的效果。 |
| `enter[].durationMs` | 0～5000。 |
| `exitNodeIds` | 要退場（隱藏）的節點 id 陣列，同樣必須跟這個 Step 的 `target.sceneId` 同一個 Scene。 |

**要讓節點一開始是隱藏、靠 `reveal.enter` 才出現的**，記得在該節點上設定 `"initiallyHidden": true`。

---

## 配色與可讀性（務必實際核對，不要憑感覺）

- 深色背景（JianBauDau 畫布預設深色）配淺色文字是一般節點的慣例；**唯獨 Connector 標籤背景接近純白，`textColor` 必須用深色**（見上方兩處提醒）。
- 有能力計算的話，用 WCAG 相對亮度公式驗證每一組 `fill`（或該節點實際可見的背景色）與 `textColor` 的對比度 ≥ 4.5:1（大標題 ≥ 3:1），不要用肉眼猜。
- **文字壓在漸層上**：漸層每一個顏色（含 `gradientStops`）都要跟文字達標，取最差的那一個。
- **文字壓在半透明填色上**：要用「半透明色疊在它底下實際顏色」的合成結果去算（`合成 = 前景 × 透明度 + 底色 × (1 − 透明度)`），底下有好幾種顏色就每一種都算。深色半透明卡片配白字比較容易達標。
- 陰影不影響對比，但黑色陰影在深色頁面底上看不出來（見「陰影」一節）。
- 避免整份文件所有卡片都用同一種 `fill`/`shapeKind`/排版——不同知識單元（或不同頁）之間刻意做出視覺差異（見 `PRESENTATION_DESIGN_GUIDE.md`）。

---

## 驗證規則與常見錯誤（產生檔案後務必逐條自我檢查）

App 匯入檔案時會執行結構驗證，以下任何一條不成立，整份檔案會被**整批拒絕匯入**（不是忽略單一節點）：

1. `schemaVersion` 必須是 `3`；`rootSceneId` 對應的 Scene 必須存在。
2. 每個 Scene 的 `camera.zoom` 必須是有限數字且 `> 0`；`camera.x`/`y`/`rotation` 必須是有限數字。
3. 每個 Scene 的 `nodeIds` 列出的每一個 id，必須在 `nodes` 裡實際存在，且該節點的 `sceneId` 等於這個 Scene 的 `id`（**雙向一致**，這是最容易犯的錯）。
4. 一個節點 **只能屬於一個 Scene**（不能同一個節點 id 出現在兩個 Scene 的 `nodeIds` 裡），也 **不能是孤兒**（存在於 `nodes` 裡，但沒有被任何 Scene 的 `nodeIds` 引用到）。
5. 每個節點的 `transform.x/y/scaleX/scaleY/rotation`、`width`、`height` 都必須是有限數字；`width`/`height` 必須 `> 0`；`scaleX`/`scaleY` 不能是 `0`。
6. `type: "frame"` 的節點必須有合法的 `frame` 內容，且 `frame.childSceneId` 指向的 Scene 必須存在、該 Scene 的 `parentFrameId` 必須等於這個 Frame 節點的 `id`。
7. `type: "image"` 必須有非空的 `image.src`；`type: "video"` 必須有非空的 `video.src`；`type: "embed"` 必須有合法（已正規化）的 `embed.url`。
8. `type: "connector"` 必須有合法的 `connector` 內容：`sourceNodeId`/`targetNodeId` 都存在、都跟這條 Connector 同一個 Scene、都不是 Connector 自己、彼此不相同；`sourceHook`/`targetHook` 都是 0～15 的整數。
9. Scene 之間透過 `parentFrameId` 往上追溯，**不能形成循環**（不會出現「A 的父層最終繞回 A 自己」）。
10. 每個 Step：`id` 非空字串、`label` ≤ 200 字、`durationMs` 落在 180～10000、`holdMs` 落在 0～30000、`easing` 是三選一合法值、`speakerNotes`（若有）≤ 20000 字。
11. 每個 Step 的 `target.sceneId` 必須對應到實際存在的 Scene；`target.kind` 必須是四選一合法值。
12. `scene-point` 的 `point.x`/`point.y`/`zoom` 必須是有限數字且 `zoom > 0`。
13. `focus-node` 的 `nodeId` 必須存在、且該節點的 `sceneId` 等於這個 Step 的 `target.sceneId`。
14. `enter-frame` 的 `frameId` 必須存在、`type` 是 `"frame"`、且該節點的 `sceneId` 等於這個 Step 的 `target.sceneId`。
15. `reveal.enter`/`reveal.exitNodeIds` 裡列出的每一個節點，都必須跟這個 Step 的 `target.sceneId` 屬於同一個 Scene（見上方 Step 一節的說明）。

**填色效果與多邊形的欄位不會讓檔案被拒絕，但不合法的值會被靜默修正或丟掉**（所以產生之後要自己確認畫面是你要的）：

- `fillOpacity`、`shadowOpacity` 會被夾進 0～1；`shadowBlur` 小於 0 會被夾成 0；非有限數字的欄位會被丟掉。
- `gradientTo` 不是非空字串 → 整個漸層不生效（`gradientType`、`gradientStops`、`gradientAngle` 也都沒作用）。
- `gradientType` 只認 `"radial"`，其他值當作線性。
- `gradientStops` 裡 `offset` 不是有限數字、`color` 不是非空字串的項目會被丟掉；`offset` 被夾進 0.01～0.99、依 `offset` 排序、最多留 8 個；整個陣列都不可用就視為沒有。
- `shadowColor` 不是非空字串 → 沒有陰影。
- `shapeKind: "polygon"` 但 `polygonPoints` 不足 3 個、超過 200 個、或含非有限數字 → **退回成一般矩形**。`smooth` 只認 `true`。

除了上面這些之外，還有幾個**不會報錯、但會讓畫面明顯出錯**的地方，agent 產生內容時要主動避免：

- Connector 的 `style.textColor` 用了淺色 → 文字在近白底卡片上幾乎看不見（見上方提醒）。
- 非矩形 `shapeKind`（`star`/`diamond`/`polygon`/... ）裡塞了長文字 → 文字被形狀邊緣截斷。
- `scene-point` 沒填 `referenceViewport`，或它的比例與 `aspectRatio` 不符 → 播放時畫面留白過多、或裁到內容一部分。
- 多頁並列時頁與頁之間沒留空隙 → 鄰頁在畫面邊緣露出一條。
- 多邊形的 `transform`／`width`／`height` 沒有貼齊頂點的範圍 → 圖形比選取框大或小，看起來位置跑掉。
- 在深色頁面底上用黑色陰影 → 看不到陰影。
- `metadata.aspectRatio` 忘了填 → 會被視為缺省值處理成 `"16:9"`，如果你真的要 `4:3` 一定要顯式填。

---

## 給 agent 的產生流程建議（Checklist）

拿到使用者的文本後，建議照這個順序進行：

1. **拆解內容**：決定用 A（單 Scene 多頁）還是 B（巢狀 Frame 知識單元）。線性、一頁接一頁的內容用 A。要用 B 時，把文本拆成幾個獨立的「知識單元」（章節/主題/概念），評估要不要有子單元（巢狀不建議超過 3 層）；跨單元的因果/比較/流程關係另外記下來，稍後用 Connector 表達，不要塞進單一單元裡。
2. **規劃 Scene 樹與命名**：A 只有 `scene-root`；B 是根 Scene + 每個一級單元的 Frame 與子 Scene，子單元依此類推。先把所有 `sceneId`/`nodeId` 命名定下來（建議語意化命名，如 `frame-<單元>`、`scene-<單元>`、`<頁>-title`）。
3. **在每個 Scene（或每一頁）裡放 Node**：頁面背景（最低 `zIndex`）、標題文字、說明文字、圖示形狀 + 標籤、Frame、Connector。同一頁內的物件彼此不重疊、留白合理；不同頁／不同單元之間的排版盡量做出差異。需要的話用漸層、半透明、陰影、多邊形增加層次。
4. **檢查配色對比**：尤其 Connector 標籤的 `textColor` 記得用深色；文字壓在漸層／半透明上要逐色計算。
5. **寫 Step 序列**：A 每一頁一個 `scene-point`（含 `referenceViewport`）；B 通常每個單元至少 2～3 個 Step——`enter-frame` 進場、一個或多個 `scene-overview`/`focus-node`/`scene-point` 展開細節（需要的話搭配 `reveal` 逐項揭示 + 對應節點的 `initiallyHidden: true`）、視需要退回上一層。所有 Step 依序串成一個扁平陣列，`order` 依序遞增。
6. **組裝成完整 JSON**：`schemaVersion: 3`、`rootSceneId`、`scenes`、`nodes`、`steps`、`metadata`。
7. **自我檢查**：逐條核對上方「驗證規則與常見錯誤」，特別是 Scene↔Node 的雙向 `nodeIds`/`sceneId` 一致性、Frame↔子 Scene 的雙向 `childSceneId`/`parentFrameId` 一致性、Connector 的節點參照與 hook 範圍、Reveal 節點與 Step 目的地同 Scene——這些是最常出錯、也最容易在自我檢查時用簡單的程式邏輯核對出來的地方。
8. **實際看一遍**：能的話用 Player（或 App 的 Present）開啟，逐頁看過——文字有沒有被截斷、陰影看不看得到、每頁是否填滿畫面。
9. 輸出最終 JSON 檔案內容（純 JSON，不要包在 Markdown code fence 之外的說明文字裡，方便使用者直接存成 `.json` 檔用「Open」匯入）。

---

## 最小可行範例（單一 Scene、單一頁，先求「能匯入、能播放」）

一頁：漸層背景、半透明＋陰影的卡片、一個曲線多邊形、標題。`scene-point` 加 `referenceViewport`，整頁填滿畫面。

```json
{
  "schemaVersion": 3,
  "rootSceneId": "scene-root",
  "scenes": {
    "scene-root": {
      "id": "scene-root",
      "name": "根畫布",
      "parentFrameId": null,
      "nodeIds": ["page-bg", "page-card", "page-blob", "page-title"],
      "camera": { "x": 640, "y": 360, "zoom": 0.7, "rotation": 0 }
    }
  },
  "nodes": {
    "page-bg": {
      "id": "page-bg", "type": "shape", "shapeKind": "rectangle", "name": "頁面背景", "sceneId": "scene-root",
      "transform": { "x": 0, "y": 0, "scaleX": 1, "scaleY": 1, "rotation": 0 },
      "width": 1280, "height": 720, "zIndex": 0, "locked": false,
      "style": { "fill": "#0f1c3a", "gradientTo": "#0e6b73", "gradientAngle": 135, "stroke": "transparent", "textColor": "#ffffff", "fontSize": 20, "radius": 0, "borderWidth": 0 },
      "text": ""
    },
    "page-blob": {
      "id": "page-blob", "type": "shape", "shapeKind": "polygon", "name": "曲線塊", "sceneId": "scene-root",
      "polygonPoints": [
        { "x": 8, "y": 40, "smooth": true }, { "x": 40, "y": 4, "smooth": true }, { "x": 86, "y": 18, "smooth": true },
        { "x": 98, "y": 66, "smooth": true }, { "x": 56, "y": 98, "smooth": true }, { "x": 14, "y": 84, "smooth": true }
      ],
      "transform": { "x": 780, "y": 150, "scaleX": 1, "scaleY": 1, "rotation": 0 },
      "width": 380, "height": 400, "zIndex": 1, "locked": false,
      "style": { "fill": "#5eead4", "gradientTo": "#3b82f6", "gradientAngle": 135, "fillOpacity": 0.7, "stroke": "rgba(255,255,255,0.45)", "textColor": "#ffffff", "fontSize": 20, "radius": 0, "borderWidth": 2,
                 "shadowColor": "#020617", "shadowOpacity": 0.45, "shadowBlur": 26, "shadowOffsetX": 0, "shadowOffsetY": 16 },
      "text": ""
    },
    "page-card": {
      "id": "page-card", "type": "shape", "shapeKind": "rectangle", "name": "半透明卡片", "sceneId": "scene-root",
      "transform": { "x": 100, "y": 380, "scaleX": 1, "scaleY": 1, "rotation": 0 },
      "width": 640, "height": 200, "zIndex": 2, "locked": false,
      "style": { "fill": "#0b1020", "fillOpacity": 0.6, "stroke": "rgba(255,255,255,0.6)", "textColor": "#eef2ff", "fontSize": 28, "radius": 28, "borderWidth": 2,
                 "shadowColor": "#000000", "shadowOpacity": 0.5, "shadowBlur": 40, "shadowOffsetX": 0, "shadowOffsetY": 24 },
      "text": "半透明卡片壓在\n漸層背景與曲線多邊形上"
    },
    "page-title": {
      "id": "page-title", "type": "text", "name": "標題", "sceneId": "scene-root",
      "transform": { "x": 100, "y": 200, "scaleX": 1, "scaleY": 1, "rotation": 0 },
      "width": 640, "height": 120, "zIndex": 3, "locked": false,
      "style": { "fill": "transparent", "stroke": "transparent", "textColor": "#ffffff", "fontSize": 64, "radius": 0, "fontWeight": 800, "borderWidth": 0 },
      "text": "**主題標題**"
    }
  },
  "steps": [
    {
      "id": "step-1", "order": 0, "label": "第 1 頁",
      "target": { "kind": "scene-point", "sceneId": "scene-root", "point": { "x": 640, "y": 360 }, "zoom": 1, "referenceViewport": { "width": 1280, "height": 720 } },
      "durationMs": 900, "holdMs": 500, "easing": "smooth"
    }
  ],
  "metadata": {
    "title": "範例簡報",
    "createdAt": "2026-09-26T00:00:00.000Z",
    "updatedAt": "2026-09-26T00:00:00.000Z",
    "aspectRatio": "16:9"
  }
}
```

更完整的範例：

- 單 Scene 多頁、填色效果與多邊形：[`examples/fill-effects-and-polygons.json`](examples/fill-effects-and-polygons.json)（7 頁）。
- 多 Scene、Frame 巢狀、Connector 關聯：[`examples/photosynthesis-knowledge-units.json`](examples/photosynthesis-knowledge-units.json)。
