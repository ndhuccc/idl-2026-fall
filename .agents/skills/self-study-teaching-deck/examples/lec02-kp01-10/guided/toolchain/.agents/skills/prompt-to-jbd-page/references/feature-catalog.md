# JianBauDau 視覺元件與效果清單(設計時逐項對照,不要只用方塊加文字)

依 `image-to-jbd-page/docs/AI_AGENT_FILE_FORMAT_SPEC.md` 整理。欄位細節與限制以該說明書為準;本表回答「這個功能適合在什麼情境用」。每份簡報要涵蓋下面大多數類別(稽核第 11 項),每個效果都要有表達目的。

| # | 功能(欄位) | 適合的情境 | 注意 |
|---|---|---|---|
| 1 | **基本形狀**:矩形(圓角)、橢圓、三角形、菱形、五邊形、星形、箭頭形狀 | 語意圖示:菱形=判斷、星形=最佳點或重點、三角形=警示、箭頭=方向、橢圓=節點或角色 | 非矩形形狀裡不放長文字 |
| 2 | **多邊形與折線**(`polygon`、`polyline`,`polygonPoints`,頂點 `smooth:true`) | 曲線、函數、區域、等高線、座標軸、波形 | 座標 0–100% 對應外框;平滑頂點用 Catmull-Rom |
| 3 | **連接線**(`type:"connector"`:16 掛點、`curve`、`direction`、`arrowSize`、`waypoints`、`label`) | 流程圖、因果、對應、網路結構。**線會跟著節點移動**,比手畫折線好維護 | 起訖節點要在同一場景;線永遠畫在最上層;標籤文字用深色 |
| 4 | **填色透明度**(`fillOpacity`、`rgba`) | 毛玻璃卡片、疊圖、高亮區域、死區或限制區 | 只影響填色,不影響邊框與文字 |
| 5 | **漸層**(`gradientTo`、`gradientAngle`、`gradientType:"radial"`、`gradientStops` 多色) | 強弱、程度、能量、溫度、立體感;徑向=光暈、中心到邊緣;多色=色帶、熱度圖 | 有 `gradientTo` 其他漸層欄位才生效 |
| 6 | **陰影與光暈**(`shadowColor/Opacity/Blur/OffsetX/Y`) | 浮起的卡片、被強調的物件(光暈)、硬邊位移的貼紙感 | 淺色底才看得到黑影;有色光暈在任何底色都明顯 |
| 7 | **線型**(`borderStyle`:`dashed`、`dotted`、`double`;`borderWidth`) | 虛線=示意、未發生、對應、限制線;雙線=強調邊界 | 沒邊框要 `borderWidth:0` 且 `stroke:"transparent"` |
| 8 | **旋轉與縮放**(`transform.rotation/scaleX/scaleY`) | 斜貼的印章、貼紙、標籤,製造手作感;鏡像 | 文字旋轉後仍要可讀 |
| 9 | **文字格式**:粗體、斜體、`$LaTeX$`、行內 `{color=#rrggbb}…{/color}`、`{size=N}…{/size}`、字體、`letterSpacing` | 在一句話裡只強調關鍵字(變色、放大),不必另開卡片;公式與文字混排 | 行內格式是否被 JianBauDau 編輯器支援,以說明書與實際播放為準 |
| 10 | **逐步出現**(`initiallyHidden` + `reveal.enter`:`fade`/`scale`/`none`,`delayMs`、`durationMs`、`timing:after-camera/with-camera`) | 一頁分 2–3 個 beat;一個接一個跳出來(用不同 `delayMs`) | 每個隱藏節點只揭示一次;節點須與步驟同場景 |
| 11 | **退場**(`reveal.exitNodeIds`) | 前一個 beat 的內容淡出,讓位給下一個(避免畫面擠);「換一個例子」「換一組參數」 | 退場節點須與步驟同場景 |
| 12 | **鏡頭運動**(`scene-point` 的 `point`/`zoom`/`rotation`、`focus-node`、`scene-overview`) | 放大聚焦某個公式或圖的一角、平移到下一區、旋轉做視角切換;**同一頁的 beat 之間移動鏡頭**,視線自然被帶到重點 | `scene-point` 要填 `referenceViewport` |
| 13 | **畫框與嵌套畫布**(`type:"frame"`、`enter-frame`) | 標題頁、章節地圖、縮圖導覽、放大看細節(畫中畫) | 只有 `enter-frame` 能換場景 |
| 14 | **群組**(`type:"group"`) | 把角色(吉祥物)或一組圖示綁在一起移動 | |
| 15 | **圖片、影片、嵌入**(`image`、`video`、`embed`) | 照片、實驗影片、YouTube | 課程 repo 的簡報不放 `asset:` 或 base64 大圖;原生簡報頁預設不貼圖 |

## 搭配示例(靈活組合,一張圖通常 3 種以上)

- **函數與誤差**:平滑折線畫曲線 + 半透明多邊形標誤差區 + 星形標最佳點 + 虛線標雜訊地板 + 行內變色強調數字。
- **流程與判斷**:菱形判斷 + **連接線**(帶標籤「是/否」)+ 退場讓上一步淡出 + 鏡頭放大到目前步驟。
- **比較兩種做法**:並排卡片用不同色系與形狀雙重編碼 + 半透明差距帶 + 多色漸層表示程度 + 箭頭標出差異。
- **重點與警示**:有色光暈的卡片 + 旋轉的印章 + `scale` 進場。
- **鏡頭帶領**:整張圖先全景(beat 1),再 `zoom` 到公式的一項(beat 2),再平移到結論(beat 3)。
