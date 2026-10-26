# deckkit — 版面流程工具(給大量 KP 簡報用)

把「頁面 = 一組卡片/表格/圖」寫成 Python,自動排版、超出就切「(續)」頁,用 `preview_deck.py` 量到的內容高度回灌,避免溢出。

檔案:`kit.py`(頁面、單元、組裝)、`flow.py`(Card/Row/Col/Table/Fig 流式版面)、`fit.py`(迭代量測)、`add_fig.py`(登錄圖)。

## 流程(在你自己的工作資料夾 W,不要放進 repo)
1. 圖腳本先產生 `assets/figures/<id>.svg`(深色)與 `.png`(淺色)。登錄:
   `py add_fig.py W <id> "<alt 文字>" <repo_root>`  → 寫入 `W/figs.json`
2. 寫 `W/deck_kpNN.py`(範本:`example_deck.py`),設 `DECK_PFX`(每個 KP 唯一,如 `lec03kp09`)。
3. `cd W; rm -f need.json; py <skill>/tools/deckkit/fit.py W deck_kpNN.py`(迭代到 grown 0,產生 `W/draft.jbd.json` 與 `W/prev/`)。
4. 稽核:`py tools/audit_deck.py W/draft.jbd.json ...`;縮圖:`preview_deck.py W/draft.jbd.json --sheets`;看 `W/prev/sheet_1.png`。
5. 通過後複製 `W/draft.jbd.json` 到 `slides/<name>.jbd.json`。

## 原生節點繪圖(優先於嵌入圖片,CLARITY C23)
由方框、長條、圓點、方塊、三角形、直線與文字組成的圖,用 `nativefig.NFig` 寫一次規格,輸出兩份:簡報用原生 JianBauDau 節點(幾 KB,文字可搜尋可編輯,放大不糊),講義用同一份規格輸出 PNG(`to_png`),數字與版面一致。
```python
from nativefig import NFig
f = NFig("lec05-kp23-dI", 540, 440, alt="圖上所有數字與結論")
f.rect(70, 232, 85, 36, fill="#5fb8a8"); f.ellipse(100, 128, 17, "#6a9fd6"); f.line(30, 150, 510, 150, "MUT", 2); f.text(270, 16, "label", "TXT", 16, True)
f.to_nodes(page, 60, y, "dark")      # 簡報;節點名稱 fig:<id>#k,區域節點 fig:<id>|<alt> 讓稽核認得是圖
f.to_png("assets/light/lec05-kp23-dI.png")   # 講義
```
規則:字級至少 16;旋轉單位是度;`TXT`、`MUT`、`GAP`、`TRACK` 隨主題變色,其餘用十六進位色;標籤用純文字(`* _ $ #` 會自動跳脫)。**畫不出來的才用 SVG 或 matplotlib**:連續曲線、數百個點的散布圖、需要數學排版的圖。稽核把 `fig:` 開頭的節點視為圖的零件,不算正文,但仍檢查字級與對比。`run_kp.sh` 要把 `*_nfigs.py` 與 `nativefig.py` 複製進工作資料夾(已處理)。

## 常用語法
`u = kit.Unit("u01", 標題, 0, 0)`;`u.flow(label, zh標題, en標題, 講稿, bridge(...), C(...), Table(...), Row(...))`;
手排頁:`p=u.page(label, zh, en)`、`p.card(key,x,y,w,h,text,c=色)`、`p.conn(a,hook,b,hook)`、`p.figure(id,x,y,w)`、`p.step(講稿)`。
色:`MAIN REF EDGE AUX WARN OK`。`T(zh, en(en_))` 組雙語行。第一個 step 的 speakerNotes 要記錄教材程度與時長。

## 單語版本(中文版、英文版分開)
在組裝程式裡用環境變數 `KP_LANG=zh|en|bi` 切換:`T()` 在 zh 版去掉英文斜體行,在 en 版去掉含中文的行並用另寫的英文覆蓋(標題、標籤、術語卡名稱 `term:<English>`、備註、圖的 alt)。各語言用各自的工作資料夾(`need.json`、`figs.json` 分開),再各跑一次 `fit.py`。範例:`lec05-kp03-polyhedral-codes-review.zip` 內的 `assets/figures/src/deck_kp03.py` 與 `i18n_data.py`。
