# 第二章自學簡報建頁來源

2026-10-30 教師核准同步至 GitHub。正式原生檔位於課程根目錄 slides/，中英文各 421 個內容頁（KP01–KP09 導學版各 376 頁，KP10 圖解版 45 頁）；既有全章地圖保留。

guided/ 保存九個 KP 共用的建頁程式與數值、逐頁提示詞／證據表、來源原生圖和驗證紀錄。diagram-kp10/ 保存 KP10 圖解版的建頁程式與檢查資料。原生來源含刻意公開的自學導讀和講義練習解析，沒有學生資料或未公開考試內容。

重建需 Python numpy、scipy、matplotlib、Pillow，Node KaTeX/marked；瀏覽器驗證需 Playwright 及 Chromium。toolchain/ 為製作時核對的工具快照，不含 node_modules。於 toolchain/.agents/skills/slides-to-html 安裝其 package.json 的依賴，再在 guided/ 執行 python3 build_guided.py 與 python3 convert_all.py；KP10 執行其 build_kp10.py。產出留在各自 output/，發布時只複製 .jbd.json 至課程 slides/。

審閱用 HTML 的 enhance_notes.mjs 提供導讀公式預排版與節點快取，僅作重現審閱版本；正式網站仍由課程原有 slides-to-html 轉換器產生，未修改共用轉檔工具或工作流程。歷史製作記錄中的「未推送」指當時草稿狀態；本 README 記錄本次同步狀態。

KP01–KP09：63 項結構／涵蓋／風格／品質檢查通過，三輪渲染後中英文共 752 頁無檢出溢位，兩種解析度驗證 4,548 步次；字型檢查 10,722 個節點無問題、無縮字。KP10：45 頁、每語 136 步，26 個來源錨點均覆蓋，檢查紀錄見 diagram-kp10/。JianBauDau App 匯入與 PDF 未實測。

啟發式警告：完整自學稿刻意放入公開備註；圖解未強求退場／頁內鏡頭移動，部分科學線條及教學道具仍被手繪箭頭規則標記，完整輸出見 validation.json，無 FAIL。

本次課程檢查器另把小數末尾的九位數標為疑似學號：KP06 是 sigmoid 值與導數，KP07 是 ln(3)，KP08 是 logistic 預測機率，KP09 是高斯反應、其加總與鄰域係數；均為算例數值。20 份的備註警告是自學導讀刻意公開，並非教師私註。
