# 課程進度與決策紀錄(COURSE_STATE)

> **給 AI agent**:每次工作開始先讀本檔,結束前更新本檔。
> 勾選 `[x]` 表示完成;不要刪除舊紀錄,新紀錄加在「決策與異動紀錄」最上方。
> 此檔會被提交到 Git,**請勿寫入成績、學號、個資、密碼或 Token**。

## 基本資料

| 項目 | 內容 |
|---|---|
| 課程 | 深度學習導論(CSIE3544AA) |
| 學期 | 2026 秋季(東華 115/1,第 1 週起始 2026-09-07) |
| Repo | ndhuccc/idl-2026-fall |
| Repo 可見度 | Private(gh 確認) |
| 使用的 agent 工具 | Claude Code(桌面版) |
| agent 能力等級 | L2(gh 已登入,可建立 Repo 與修改設定) |

## 建立學期的檢查表(AGENTS.md §3)

- [x] Step 1 老師有 GitHub 帳號(ndhuccc)
- [x] Step 2 用範本建立學期 Repo(2-A 自動:gh repo create --template ndhuccc/course-template --private)
- [x] Step 3 agent 取得 Repo 存取(clone 到 C:\Users\ccchiang\Projects\idl-2026-fall)
- [x] Step 4 完成訪談,產生 `semester.toml`
- [x] Step 5 套用設定;網站建置無 ERROR;starter 19 失敗、solution 19 通過(Lab 03 因未裝 PyTorch skipped)
- [x] Step 6 提交並推送
- [x] Step 7a 啟用 Pages(Source: GitHub Actions),網址 https://ndhuccc.github.io/idl-2026-fall/
- [x] Step 7b 開啟 Discussions
- [x] Step 7c 手動觸發 `deploy-site` 並確認網站可開(run 38339534218 成功,網站回應 200)
- [ ] Step 8 初始化內容:週次(完成)、課綱評分與政策(完成)、範例 Lab 已刪除;HW 與講義內容待補(見待辦)
- [ ] Step 9 向老師總結

## 待辦事項

- [ ] HW 01–03 的確切發佈與截止日;`assignments/hw01/README.md` 目前仍是範本示範作業(梯度下降),要換成實際的 HW 01「反向傳播與 MLP」(依 `.agents/skills/assignment-design/SKILL.md`)
- [ ] 期末專題確切截止日;e 學苑繳交說明
- [ ] LabTube:在 LabTube 建課、匯入修課名單,並把「學生作業 Repo 名稱前綴」設為 `idl-2026`;向學生公告作業範本網址與 LabTube 網址
- [ ] 之後修改 `student-template/` 時,要手動同步到 `ndhuccc/idl-2026-hw-template`(學生已建立的 Repo 不會自動更新)
- [ ] 每週講義與投影片(第1週KP01–KP10與第3週KP01–KP16已完成，其餘待補)
- [ ] 考試範圍(期中第 9 週、期末第 17 週)

## 決策與異動紀錄

- 2026-10-16 — 講義標題格式再統一:所有講義(含導論與整份課堂講義)一律「KPNN 標題」,每週編號從 KP01 起依進度表順序遞增;`scripts/sync_nav.py` 的 `check_titles` 同步擴大為檢查所有講義與週內編號。pr-2026-fall:第 1 週導論改為 KP01;第 2 週不再區分「正課」與「自主學習」,九份講義編為 KP01–KP09 並依內容拆成三列(貝氏決策與線性分類器 KP01–KP02、機率密度估計 KP03–KP06、單純貝氏最近鄰居與貝氏網路 KP07–KP09),各列附講義與簡報連結;原「選讀,不列入考試範圍」的備註隨之移除(考試範圍需另行確認)。course-template 的 lec01 連結文字改為「KP01 課程介紹與環境設定」。AGENTS.md §5.12 第 1、6 項更新,三個倉庫相同。
- 2026-10-16 — 建置腳本自動檢查講義標題格式:`scripts/sync_nav.py` 新增 `check_titles`,每次建置(`build_docs.py` 會呼叫它)檢查選單用的講義標題——知識點講義須為「KPNN 標題」且編號與檔名一致、不得有「(試做)」等標註、不得附英文括號、須列入 schedule.md(否則落到「其他」);有問題只顯示「[警告] 講義標題格式」,加 `--strict` 才以非 0 結束。AGENTS.md §5.12 第 6 項補上說明。course-template 的 `lectures/lec01.md` 尚未列入進度表,會出現一則警告(範本狀態,課程設定後消失)。三個倉庫相同。
- 2026-10-16 — 統一各課程左側選單的講義標題格式:知識點講義一律 `KPNN 標題`(KP 加兩位數編號、一個半形空格、中文短標題,不用冒號、不加「(試做)」等標註、不附英文);規則寫入三個倉庫 `AGENTS.md` §5.12 第 6 項。pr-2026-fall 的 `schedule.md` 第 3 週 KP01–KP18 連結文字依此改寫(原本混用「KP01：」與「KP02 」),idl-2026-fall 原本即符合。
- 2026-10-16 — 依老師要求把「週次進度表跟著實際製作的講義內容更新配置」寫進標準工作規範:`AGENTS.md` 新增 §5.12(依實際講義拆列並概括主題、每列列出存在的講義與簡報連結、備註寫知識點範圍、未列入進度表的講義會落到選單「其他」、不自行改週次日期、驗證與記錄),§5.1、§5.11 的同步步驟指向 §5.12,§9 檢查清單新增一項。pr-2026-fall、course-template、idl-2026-fall 三個倉庫的 `AGENTS.md` 相同。
- 2026-10-16 — 新增技能 `.agents/skills/slides-to-four-panel-comics/`(讀簡報、擷取每個核心概念、為每個概念產生一張 16:9 的 2×2 四格漫畫並插入簡報;含 `SKILL.md`、`references/jbd-format.md`、`scripts/insert_jbd_comics.py` 等);三個倉庫(idl-2026-fall、pr-2026-fall、course-template)內容相同。
- 2026-10-14 — 第二章 KP01–KP10 驗收通過並已推送(commit c35b244)。新增全章地圖 `slides/lec02-kp00-chapter-map.jbd.json`(1 頁,十個 KP 蛇行排列,公式用 LaTeX),本機版尚未推送。技能 `audit_deck.py` 新增 G3(公式須用 LaTeX、不得用 Unicode 近似字元),已同步到 course-template 與 pr-2026-fall;第 2 章 KP01–KP10 簡報仍有 Unicode 公式,老師決定這個單元先不改。

- 2026-10-14 — 第二章 KP08–KP10 本機版(尚未同步雲端):老師驗收 KP05–KP07 後接續製作。KP08 ADALINE 與邏輯回歸(OR 手算 LMS、牛頓法一步)、KP09 RBF 與 SOM(XOR 的 RBF 解、SOM 一步更新)、KP10 萬有逼近定理(階梯法逼近 sin 2πx、誤差≈π/N)。講義與簡報(各 4 頁)已通過稽核。來源更正:牛頓更新應為加號且對數似然是凹的;SOM 鄰域逐漸縮小而非每步對半;RBF XOR 例需偏置;UAT 對 ReLU 的補充。第二章章節地圖 `lec02-kp00-chapter-map` 待做。

- 2026-10-14 — 第二章 KP05–KP07 本機版(尚未同步雲端):老師驗收 KP02–KP04 後接續製作。KP05 多層網路與非線性(2-2-1 XOR 網路、全線性坍縮)、KP06 活化函數(13 種分四族、softmax 手算、ReLU 家族)、KP07 損失函數(MSE/MAE/Huber、交叉熵/KL、hinge)。講義與簡報(各 4 頁)已通過稽核。來源更正:hinge 為零條件是 y·f≥m(來源寫 |f|≥m)。KP05 的 xor-network 圖在簡報改用原生節點與連線,稽核以 --exclude-fig 排除(待老師確認)。KP08–KP10 與 kp00 章節地圖待做。

- 2026-10-14 — 第二章 KP02–KP04 本機版(尚未同步雲端):老師驗收 KP01 後接續製作。KP02 猜錯才改權重(OR 從全 0 的 5 次更新得 (1,1,−1))、KP03 ±1 標籤的更新(代數四種組合與幾何法向量/距離兩路得 2ηyx,η=½ 軌跡與 KP02 相同)、KP04 單層限制(引用第 1 章 KP08 的 XOR 證明,不重證;實跑每輪更新 3,3,4,4… 永不歸零、格點搜尋最多對 3/4、兩條零誤差邊界對新點判斷相反)。講義三份 audit_lecture FAIL 0(KP04 的 L10 警告只來自照錄的來源自我檢核題);簡報 4+3+3 頁,audit_deck FAIL 0,預覽溢出 0,check_slides 只有講者備註公開提醒;新增圖 4 張與程式 `assets/figures/src/lec02-kp0204-figures.py`。來源更正:幾何路徑的損失只對錯分點求和、兩條路差一個係數(併進學習率)、Minsky/Papert「單獨造成寒冬」依第 1 章 KP08 改為過度簡化。待老師驗收後做 KP05–KP07。

- 2026-10-14 — 第二章(感知器與多層感知器)開工,來源 CCC-LNS `IDL_115/Ch02-perceptron-mlp`(2026-10-14 讀取)。老師確認:大學生、全章3小時(淨150分鐘)、10個KP、每KP15分鐘,中英對照,沿用第1章檔名與格式(`lec02-kpNN-<代號>`、`kp00` 全章地圖);重複處理:第1章 KP08 已講過 M-P 邏輯單元、XOR 四條不等式與 Perceptron/ADALINE 單點更新比較,第2章只引用不重證。**先前誤把這章做在 pr-2026-fall,已還原(55701af),成果備份後搬來這裡重做。** 試做 KP01(本機版,尚未同步雲端):講義 `lectures/lec02-kp01-biological-neuron-and-mp-unit.md`(audit_lecture FAIL 0/WARN 0)、簡報 4 頁(audit_deck FAIL 0/WARN 0,約14分鐘,預覽溢出0,check_slides 只有講者備註公開提醒)、圖 3 張與程式 `assets/figures/src/lec02-kp01-figures.py`。來源更正:NOT 閘改用負權重 `[補充]`;M-P「等權固定」與式(1)可學權重的矛盾已說明;互動示範 $z>	heta$ 與式(1)統一為 $s\ge0$。自第2章起預定的更正:Logistic 牛頓更新式來源寫減號應為加號、hinge 損失為 0 的條件應是 $yf\ge m$、SOM 鄰域分段縮小、RBF 中心為手選。待老師驗收 KP01 後再做 KP02–KP10 與全章地圖。

- 2026-10-14 — 第一週講義粗體標記修正：檢查KP01–KP10，CommonMark發現KP05–KP10共42段標籤因冒號放在粗體標記內且緊接文字而未解析。比照KP01–KP04，將六份講義的標籤冒號移到粗體標記外，共正規化48段（含六處原已可解析的來源標籤）；移除格式星號後的正文逐字相同，公式與數值未更動。十份講義的248段粗體均經CommonMark與本機網站HTML核對，殘留雙星號0；GitHub Markdown預覽確認KP05的25段全數解析，瀏覽器確認字重700。網站建置成功，本次同步六份講義與本紀錄。

- 2026-10-14 — KP05–KP10標題與KP07公式修正同步完成：修正版本ea8b8c91d323c36554cbad3b4323bb3a37f6f071，課程網站更新工作38552367531成功。八個修正檔案已從GitHub讀回核對，六個講義頁各只有一個中英對照主標題，線上MathJax設定與產生程式一致。線上瀏覽器確認KP07的68個公式正常，未定義命令、公式錯誤與頁面錯誤0，boldsymbol的beta/mu/varepsilon均呈粗體。

- 2026-10-14 — 依老師要求統一KP05–KP10講義標題：比照KP01–KP04，改為單一「第1週 KPnn：中文（English）」標題。KP07的boldsymbol渲染問題已在線上瀏覽器重現；MathJax將命令顯示成紅色未定義文字，公式本身不需改寫。已先說明共用設定修改原因，再於scripts/build_docs.py的MathJax設定明確載入[tex]/boldsymbol並啟用boldsymbol套件；頁面重排改在MathJax初始化完成後執行，修正延遲載入時clearCache過早執行的錯誤。本機網站建置成功，10份講義標題格式一致；瀏覽器確認KP07的68個公式均顯示，每式僅一個主要輸出，未定義命令、公式錯誤及頁面錯誤0，beta/mu/varepsilon粗體探測通過。繁體中文/英文內容、係數與數學符號保持原意，推送後再以實際網站渲染核對。

- 2026-10-14 — 第一章全部最新版已推送：教材版本c0dc909c6af29ab2cbfc6b5f25b2889b4c9b0902，雲端84個第一章教材/連結檔案物件與本機已儲存版本一致，課程網站更新工作38548052212成功。接續依老師要求，KP05–KP10講義標題移除「補充講義」，英文 supplement 改為 lecture；本文的補充例子來源標記保留。這次名稱調整同步六份講義與本紀錄，簡報內容保持老師指定的最新版。

- 2026-10-14 — 第一章全部KP最新版同步：老師將範圍擴大為KP01–KP10的最新版講義與簡報，並要求接續未完成的推送。採用老師目前本機的KP02–KP10簡報（保留新增漫畫圖片、放大鏡頭及文字/連線編輯），KP01沿用既有最新版；新增KP08–KP10講義、全章地圖、9組PNG/SVG與可重現圖程式，連同第1週連結、導覽與進度紀錄同批同步。11份第一章JSON的發布格式檢查FAIL0，提醒僅已授權公開的講者備註；9份有新編輯的簡報在播放器1.0.1的1280×720與1920×1080逐步檢查，公式、圖片載入與文字框錯誤0。老師的場景外圖片和放大鏡頭保留原版，新增放大步驟未加講者備註，未按技能的固定頁面規則重新排版；因此不把草稿版式稽核宣稱為全數通過。只同步第一章教材與必要連結，兩個技能壓縮檔與其他章教材不納入。

- 2026-10-14 — 第一章KP08–KP10及全章地圖本機版：依已確認大學生、全章3小時、淨150分鐘、每KP15分鐘的中英對照方案，完成3份講義、3份可編輯JianBauDau簡報（4＋4＋3＝11頁）、1頁全章地圖、9組PNG/SVG與可重現圖程式。講義保留來源每KP4題及推導，簡報每KP選1題加提示不標答案，每頁附公開講者備註。KP08以XOR四點分清線性表達限制與Perceptron/ADALINE更新；KP09用受控例比較共享卷積與全連接參數，區分1989早期CNN與1998 LeNet-5、2006兩篇論文及後續架構；KP10用回饋、輸出與評分串接全章。講義稽核三份FAIL0/WARN0，簡報三份FAIL0（N6僅論文DOI/arXiv識別碼）；12頁近似預覽溢出0，播放器1.0.1在1280×720與1920×1080逐頁檢查，裁切/缺失/公式/圖片錯誤0，並依實際畫面保留底部控制列空間。第一週連結與下載頁已本機更新；既有老師編輯的KP02–KP07簡報及技能壓縮檔保留，未覆寫。本批先交本機預覽，尚未同步雲端。

- 2026-10-14 — 第一章KP05–KP07已同步雲端：依老師指示，只儲存並同步該批3份講義、3份簡報、9組PNG/SVG、圖程式及第1週連結/進度。版本4a1e4fe254ef89c9dc9efe17ca93faae1b7ed5f5，課程網站更新工作38534347086成功；3份線上簡報逐檔讀回與該版本位元組一致。本機另有老師編輯的KP02–KP07簡報與兩個技能壓縮檔，均未納入該次同步。

- 2026-10-14 — 第一章KP05–KP07教材本機版：老師要求接續製作後面的知識點，依已確認大學生程度、全章3小時、中英對照與每KP15分鐘精簡配置，完成3份新講義、3份獨立JianBauDau簡報（4＋4＋3＝11頁）、9組PNG/SVG與可重現圖程式。每份講義保留來源4題自我檢核；投影片各選1題附提示、不標答案，且每頁有公開講者備註。KP05區分側資訊與互動獎勵，補充同8點PCA/FDA對比及來源兩步回報；KP06以原參數重算蘋果/黃瓜高斯密度、證據與後驗，明列來源數字錯誤；KP07補充Ridge/Lasso獨立二次例及同共變異數下FA/pPCA雜訊對比，修正JL、EM與KDR年份概括。圖與講義、簡報逐一核對，3份簡報格式稽核FAIL0；11頁近似預覽溢出0，JianBauDau Player 1.0.1於1280×720及1920×1080逐頁檢查，缺失/裁切/公式/圖片錯誤0。保留既有KP01–KP04和老師正在編輯的KP02–KP04簡報；第一週進度表新增本批連結，沒有恢復已移除的舊lec01。先交本機預覽，尚未同步雲端；KP08–KP10待本批驗收後接續。

- 2026-10-14 — 依老師指示移除第1週舊範本講義 lectures/lec01.md，刪除schedule.md的lec01連結，導覽由build_docs.py自動更新；第1週改列KP01–KP04講義，四份簡報及其附圖均保持原內容。網站本機建置成功，舊lec01輸出頁已不存在，doctor FAIL0。僅本次刪除/連結/導覽/進度4檔納入同步雲端；兩個既有技能壓縮檔留本機。

- 2026-10-14 — 第一章KP01–KP04同步完成：教材版本b890a7c，GitHub雲端33檔物件與本機已儲存版本一致，課程網站更新工作38519796229成功。已從線上讀回4份JianBauDau簡報與20個PNG/SVG，逐檔位元組與已同步版本一致；4個講義頁及第1週進度表/下載頁均含正確下載連結。KP01包含老師新增教學圖與步驟，其他章與共用腳本未改。根據範圍只同步第一章教材及相關連結/紀錄；作業期間新出現的 .agents/skills/diagram-design.zip 與 view-from-text.zip 保留本機，未納入本次同步。KP05–KP07仍待本批驗收後製作。

- 2026-10-14 — 老師授權「推送到github」。同步第一章KP01–KP04：4份講義、4份JianBauDau簡報、10組PNG/SVG、2份圖程式及第1週連結/導覽/進度紀錄，共33檔。KP02–KP04共10內容頁；KP01保留老師新增的3張教學圖與3步驟。來源數值/公式/排版查核沿用本批通過結果，新增圖片逐張檢視未含個資；四份簡報格式FAIL0，WARN僅已批准公開的講者備註。登入帳號ndhuccc、課程資料夾Private，本機與既有雲端基準一致；網站先前建置成功。本次以明確檔案清單儲存並同步，其他章與共用腳本未變；雲端與線上下載讀回待完成後再記錄。

- 2026-10-14 — 老師驗收第一章KP01並授權接續KP02–KP04。已完成本批3份大學生程度中英對照講義、3份獨立可編輯JianBauDau簡報（4+3+3＝10內容頁）、7組PNG/SVG及可重現圖程式 assets/figures/src/lec01-kp0204-figures.py。沿用全章3小時、淨150分鐘、每KP15分鐘精簡配置，來源讀取2026-10-14，保留來源數據與所有4題檢核於每份講義，不標練習答案。KP02澄清矩陣方向、p與參數數目、非單位向量投影及仿射映射，完整12點輸出最大16；KP03明示補充b=-5控制，分開有限類別/實數/內部機率，不把無單位分數稱作直徑；KP04依原15點與種子更正m=3群人數3/6/6，原5點距離表重新核算，補充z=x1顯示15樣本落10位置而非刪樣本。講義稽核均FAIL0/WARN0；簡報FAIL0，2/5/6個N6提醒分別為原生/分行等價公式及精簡主線取捨，逐項見本機FINAL-REVIEW-B.md。全部10頁近似預覽溢出0；播放器1.0.1於1280×720與1920×1080逐頁核對，裁切/缺失/公式/圖片錯誤0，嚴格解析46個公式片段；已修正文字框與圖例遮點。原生節點覆蓋KP02矩陣/映射與KP03任務對照，image-only圖稽核的排除僅為工具限制，內容未省略。逐張圖與頁面自讀完成；真實播放器採隱藏獨立設定DOM檢查，未取得原生播放器截圖，觀感與PDF匯出待老師驗收。第1週講義與下載連結已增加；KP01講義、附圖與圖程式8檔雜湊保持一致；KP01簡報於作業期間另有編輯更新（新增3個圖片節點及3個步驟），保留最新本機版，未覆蓋；其他教師教材保留。本機網站建置成功，3份下載JSON與來源檔位元組一致，預覽10頁、3單元切換及備註操作檢查通過；本批交付本機預覽，尚未同步雲端；KP05–KP07待本批驗收再開始。

- 2026-10-14 — 老師確認第一章盤點（大學生、3小時，10個KP，淨授課150分鐘，每KP15分鐘，精簡3–4頁，依序製作），先試做KP01「選模型要有評分標準」。新增 lectures/lec01-kp01-model-selection-framework.md、slides/lec01-kp01-model-selection-framework.jbd.json（4內容頁、1Frame、5Step）與3組PNG/SVG及圖程式 assets/figures/src/lec01-kp01-figures.py。主線依假設/評分/搜尋→來源A/B懲罰算例→泛化限制→完整四選項練習；原4題檢核均保留於講義，不附答案。來源主文與教授備課資料讀取2026-10-14；澄清argmin不是搜尋演算法、低訓練分數不能保證泛化、No Free Lunch平均條件與來源比喻的高/低分矛盾。新增補充為訓練/驗證/測試角色及上述精確條件，引用已核對的作者論文與官方課程資料。兩張講義概念圖在簡報以原生節點重建（流程3節點2Connector；已知/未見資料2卡無保證箭頭），圖內容未省略，影像型O1工具需排除這2個image參照並另作人工核對。講義稽核FAIL0/WARN0；簡報稽核FAIL0，保留2提醒：來源習題選項按要求沿用、2007.10928是文獻識別碼而非待展示數值。近似預覽4頁溢出0；播放器1.0.1在1280×720與1920×1080各測4頁、6停點與1次返回，文字裁切/缺失/公式/圖片錯誤0，17個公式片段嚴格解析成功。隱藏視窗測試使用獨立設定與只改時間的副本，沒有取得實際播放器截圖；觀感、SVG縮放與PDF匯出待老師驗收。第1週連結已加，保留原lec01與全部第3章老師版本；本次先交本機試做，網站本機建置成功，講義/附圖/下載檔讀回一致，近似預覽換頁與備註展開通過，尚未同步雲端；老師驗收後才展開KP02–04。

- 2026-10-14 — 依老師要求改版網站的投影片下載頁:`scripts/build_docs.py` 的 `write_slides_index` 改為一週一個頁籤、一個頁籤一張表(教材、格式、下載),教材欄顯示對應講義的標題;`mkdocs.yml` 新增 `pymdownx.tabbed`(`alternate_style: true`);`slides/FORMATS.md` 補一句說明。idl-2026-fall、course-template、pr-2026-fall 三個倉庫使用相同設計。
- 2026-10-14 — backpropagation新版教材雲端同步完成：內容版本ab8c011，課程網站發佈工作38455448883成功。線上講義 https://ndhuccc.github.io/idl-2026-fall/lectures/lec03-backprop-optimization-v2/ 、第3週進度及下載頁皆含50頁課堂版與360頁完整版連結；30張講義附圖已逐檔比對。兩份下載檔與已同步版本位元組完全一致，且與本機JSON解析內容相同（Windows換行經標準化）：課堂版SHA-256 79a9a61e0294c59c90052dd777b024a051e47d851c4c58d44f3cc2bfb05ac096；完整版SHA-256 45455b024c5b931437055769ce60710583a4247e8015977f71abb63ebf2fe09a。精準範圍同步70個教材/連結/紀錄檔，網站建置與結構檢查通過；49個既有草稿/人工修改檔雜湊未變，其他本機紀錄與導覽草稿保留未上傳。實際觀感、SVG縮放與PDF匯出仍待老師確認。

- 2026-10-14 — 老師授權「同步雲端」。本次同步backpropagation全章新版講義、50頁課堂版、360頁完整版、對應附圖/繪圖程式及第3週連結；保留其他本機草稿與人工編輯。結構檢查無FAIL、講義公式全對應、360頁近似預覽溢出0，播放器兩種尺寸裁切已修正並重檢。本次內容將由既有課程網站發佈流程更新，線上讀回與下载雜湊尚待確認。

- 2026-10-13 — 依老師指定「大學生、4小時」完成backpropagation第3章KP01–16重製：新增 lectures/lec03-backprop-optimization-v2.md、slides/lec03-backprop-optimization-v2.jbd.json（50主線＋310延伸＝360內容頁，16個Frame、376步）及課堂版 lec03-backprop-optimization-v2-core.jbd.json（50頁、66步）。規劃淨授課200分鐘、休息40分鐘，主線估178分鐘。30張不同講義附圖包含21張數值圖、3張依投影片原生節點/Connector生成的流程圖、6張來源原圖；所有簡報圖片已內嵌，講義圖全部本機可讀。完整保留來源範例、公式與練習選項/提示，明列來源更正及補充，不附練習答案。195個講義公式全對應，427個數學片段嚴格解析錯誤0，結構及內容稽核FAIL 0、360頁近似預覽最終溢出0。播放器1.0.1在1280×720與1920×1080各檢查360頁、392停點及16次返回；發現3頁4個小幅裁切後補高並重檢。測試使用獨立設定、縮短動畫及隱藏視窗定時回呼，實際觀感、SVG縮放與PDF匯出仍待老師確認。保留N3假設輸入0.0008及N6圖中數值/文獻識別碼的提醒並寫明理由；備註會隨下載公開。第3週加入新版講義及兩個簡報連結，網站本機建置成功。保留全部舊版、備份與人工修改，未修改共用腳本/技能/自動化；此階段交付本機預覽，尚未提交或同步到雲端。

- 2026-10-13 — 依老師要求啟動 backpropagation 全章講義與簡報重製。已讀更新後 `view-from-text` 與 `LEVELS.md`，讀取 CCC-LNS 第3章現行主文、16個 KP、教授祕笈、互動數值例與附圖，完成暫存盤點表。依新技能 §1 第10–12點，待老師確認知識單元表、教材程度與簡報時長後才開始批次製作；暫提研究生程度、6小時（淨300分鐘，每KP18.75分鐘，精簡主線），主線估74頁、延伸估200頁。擬新增整章 `lec03-backprop-optimization-v2` 講義及簡報，保留既有版本與人工修改。此階段尚未製作新講義或簡報，也未同步到雲端。

- 2026-10-13 — 依老師要求更新課程網站的 JianBauDau 播放器下載連結與顯示版本為 1.0.1，指向 `ndhuccc/IDL115` 的 `player-v1.0.1` 公開下載檔；同步投影片格式說明。本機網站建置成功，doctor 無 FAIL；透過既有課程網站發佈流程更新。

(最新的在最上面。格式:`YYYY-MM-DD — 決定或異動內容`)

- 2026-10-14 — 依老師指示刪除舊版 backpropagation v2 教材:`lectures/lec03-backprop-optimization-v2.md`、`slides/lec03-backprop-optimization-v2.jbd.json`、`slides/lec03-backprop-optimization-v2-core.jbd.json`,以及 `assets/figures/lec03-bp-v2-*` 與 `assets/figures/src/lec03-bp-v2-*`(Git 歷史仍保留)。`schedule.md` 第 3 週列改為連到 KP01–KP16 十六份講義與全章地圖加十六份簡報,`mkdocs.yml` 導覽由 `build_docs.py` 重新產生;捨棄先前未提交、連到已不存在檔案(lec03-kp01-optimization-introduction 等)的導覽草稿。`mkdocs build` 成功,第 3 章十六頁講義與簡報下載頁都產生,講義連結的圖片與簡報檔都存在。老師已在 JianBauDau 修改 KP01–KP16 的簡報並要求保留修改版:KP01–KP05 已提交;KP06–KP16 本機版本為老師修改版,KP06 的第一頁銜接條就地改為「KP01–KP05 都把梯度當成已知量」。雲端網站要重新發佈後才會移除 v2 頁面。
- 2026-10-14 — 批次 F 由主代理依序完成,第 3 章 16 個知識點全部以新版技能重做完畢:KP15 過參數化網路的收斂保證(4 頁)、KP16 全章總結(4 頁)與全章地圖頁(1 頁,獨立檔 `slides/lec03-kp00-chapter-map.jbd.json`,原生節點加連線,五個階段);講義 `lectures/lec03-kp15-overparameterized-convergence.md`、`lec03-kp16-chapter-synthesis.md`,圖程式 `assets/figures/src/lec03-kp15-figures.py`(3 張圖,含「n=3、p=10 的最小平方」「d=2、n=3 二次啟用網路寬度實驗」「p=50 兩筆樣本 δ 與步數」)與 `lec03-kp16-figures.py`(1 張圖,15/15/2000 筆的多項式訓練/驗證/測試誤差)。稽核:KP15、KP16 簡報 FAIL 0(WARN 為 T1 時間估計與 N6),講義 FAIL 0 WARN 0。整合:列出 16 份簡報的標題鏈與銜接條逐頁讀過,修正 KP06 第一頁(「KP01–KP05 都把梯度當成已知量」)與 KP10 第一頁(接 KP09 留下的「一個全域學習率顧得了所有維度嗎?」)的銜接條。來源更正重點:不把「所有局部極小等於全域極小」推廣到一般網路(只在定理條件下,且 k=1 時零損失不存在)、「O(log 1/ε)」稱線性收斂要說明是對精度的對數依賴、Proposition 5 的網路型態(KP15 說全連接 ReLU,KP16 說殘差)統一依 KP15、刪除「六階工具鏈」說法改用問題對工具表、§7.3 文獻條目未逐篇核對只當作來源列舉的方向。寬度實驗發現:k≥4 沒到零損失的 5 次是步長太大發散(改小步長全部到達),不是假局部極小。未處理:KP01–KP05 的本機簡報檔被編輯器重新匯出過(多出步驟與圖片 alt「image」),稽核會出現 FAIL,尚未決定保留或還原;v2 講義與簡報、`lec03-bp-v2-*` 圖待老師決定;`schedule.md` 與導覽未加入第 3 章各 KP;技能新功能尚未同步到 course-template 與 pr-2026-fall。
- 2026-10-14 — 批次 E 用 3 個子代理平行處理(KP12 平坦極小值與 SAM 動機、KP13 基本 SAM、KP14 高效與自適應 SAM,各 4 頁共 12 頁;大學生、4 小時):講義 `lectures/lec03-kp12-flat-minima-and-sam-motivation.md`、`lec03-kp13-vanilla-sam.md`、`lec03-kp14-efficient-and-adaptive-sam.md` 與同名簡報,圖程式 `assets/figures/src/lec03-kp12-figures.py`、`lec03-kp13-figures.py`、`lec03-kp14-figures.py`(8 張圖)。主代理重跑稽核:三份簡報 FAIL 0(WARN 為 T1 時間估計約 16 分、N6 講義數字未上簡報)、三份講義 FAIL 0 WARN 0;KP14 最後一頁補上「留給 KP15」並重排為 4 頁。例子設定:KP13 玩具碗 η=0.02、ρ=0.5、起點 (10,2),KP14 同碗同 ρ(60 步實驗 η=0.05)。來源更正重點:尖銳解損失上升為二次成長而非指數、Newton 法與大 batch 必陷尖銳改為有條件的經驗傾向、SAM 不保證平坦解(碗上 85 步後被 GD 超越)、忽略 dε*/dw 的誤差 0–11%、SWP 只有 ε 期望無偏而梯度不保證、ASAM 縮放條件僅在 γ=0 精確成立、刪除 SDS「省 50–70%」。未查證的引用(Keskar、Hoffer、Goyal、Dinh、ESAM、ASAM、SAM 的 ρ 搜尋範圍)標為 [補充]。
- 2026-10-14 — 批次 D 改用 3 個子代理平行處理(KP09 SGD 收斂 4 頁、KP10 AdaGrad 4 頁、KP11 RMSProp 與 Adam 5 頁;大學生、4 小時,共 13 頁):講義 `lectures/lec03-kp09-sgd-convergence.md`、`lec03-kp10-adagrad.md`、`lec03-kp11-rmsprop-and-adam.md` 與同名簡報,圖程式 `assets/figures/src/lec03-kp09-figures.py`、`lec03-kp10-figures.py`、`lec03-kp11-figures.py`(共 8 張圖)。新增簡報產生工具 `.agents/skills/view-from-text/tools/deckkit/`(kit、flow、fit、add_fig 與說明)。主代理重跑稽核:三份簡報 FAIL 0(WARN 為 T1 時間估計 16–18 分、N6 講義數字未上簡報,皆說明)、三份講義 FAIL 0 WARN 0,縮圖無溢出。來源更正重點:全梯度下降界用 ‖w0−w*‖²/(2ηt)(來源常數對 η 較小時不成立)、固定步長 SGD 殘差平台正比於 η(帶寬正比 √η)、1/√k 步長不滿足 Ση²<∞、刪除 AdaGrad「有效步長衰減到 10⁻⁷」說法、Adam 不做偏差修正時步長是被放大(第 1 步 3.16 倍)而非停擺、Adam 改稱「常用預設之一」。KP10 用 η=1、KP11 用 η=0.5(RMSProp 第一步會放大約 3.16 倍)。
- 2026-10-14 — 依新版技能依序完成批次 C(KP06 反向傳播 5 頁、KP07 SGD 4 頁、KP08 小批次與 epoch 4 頁;大學生、4 小時,共 13 頁):全新撰寫講義 `lectures/lec03-kp06-backpropagation.md`、`lec03-kp07-sgd.md`、`lec03-kp08-mini-batch-and-epoch.md`(各附練習、完整解題示範與來源更正)與同名簡報,圖程式 `assets/figures/src/lec03-kp0608-figures.py`(3 張圖)。稽核:三份簡報 `audit_deck.py --lecture --speaker-notes --level undergrad --narrative --minutes --source` FAIL 0;三份講義 `audit_lecture.py --source` FAIL 0、WARN 0。來源更正重點:反向傳播不是梯度下降(只算梯度)、SGD 雜訊幫助逃出鞍點等說法不採為結論、批次方差 σ²/b 僅對獨立抽樣成立(無放回要乘有限母體修正)、刪除「GPU 利用率低於 5%」與「全批次易收斂到尖銳極小值」的無依據說法。v2 講義與簡報待整章完成後再處理。

- 2026-10-14 — 依新版技能依序完成批次 B(KP03 停止準則、KP04 線搜索與最陡下降、KP05 動量;大學生、4 小時、各 4 頁共 12 頁):全新撰寫講義 `lectures/lec03-kp03-stopping-criteria.md`、`lec03-kp04-line-search.md`、`lec03-kp05-momentum.md`(各附練習與完整解題示範、來源更正)與同名簡報,圖程式 `assets/figures/src/lec03-kp0345-figures.py`(5 張圖)。稽核:三份簡報 `audit_deck.py --lecture --speaker-notes --level undergrad --narrative --minutes --source` FAIL 0;三份講義 `audit_lecture.py --source` FAIL 0、WARN 0。已刪除舊的 KP01 與 KP03 檔案(講義、簡報、圖、腳本)。來源更正重點:動量在二次函數上把穩定上限由 2/λmax 放寬為 2(1+α)/λmax(來源「動量不會讓不安全的學習率變安全」過於籠統)、線搜索回溯成功條件為 η ≤ 2(1−c)/L。

- 2026-10-14 — 依新版技能試做第 3 章 KP02(大學生、4 小時、經老師同意用 5 頁):全新撰寫講義 `lectures/lec03-kp02-gradient-descent-and-learning-rate.md`(含兩張圖、誤差倍率表、w² 例子、解題示範與來源更正)與 5 頁簡報 `slides/lec03-kp02-gradient-descent-and-learning-rate.jbd.json`,圖程式 `assets/figures/src/lec03-kp02-lr-figures.py`。稽核:`audit_deck.py --lecture --speaker-notes --level undergrad --narrative --minutes --source` FAIL 0;`audit_lecture.py --source` FAIL 0、WARN 0。已刪除舊的 KP02 檔案(另一個 session 的講義、簡報、備份與圖,以及 `-undergrad` 版)。

- 2026-10-14 — 依新版技能試做第 3 章 KP01(大學生、4 小時):老師確認盤點後,全新撰寫講義 `lectures/lec03-kp01-optimization-as-search.md`(含兩張圖、成本表、解題示範)與 4 頁簡報 `slides/lec03-kp01-optimization-as-search.jbd.json`(每頁有銜接條、講者備註),圖程式 `assets/figures/src/lec03-kp01-search-figures.py`。稽核:`audit_deck.py --lecture --speaker-notes --level undergrad --narrative --minutes 200 --kps 16 --source` FAIL 0;`audit_lecture.py --source` FAIL 0、WARN 0。取代舊的 `lec03-kp1-parameter-search.jbd.json`(已刪除)。尚未加入 `schedule.md`/導覽。

- 2026-10-14 — 老師要求把盤點報告的格式與批次處理寫進技能:新增 `.agents/skills/INVENTORY_FORMAT.md`(八部分固定格式:設定、章級知識地圖、備課筆記、頁序與銜接表、講義與檔案、批次建議與子代理、補充與取捨、請你確認或調整)。`view-from-text` 與 `textbook-lecture-notes` 動手前都要交此報告;整章不一次做完,第一批試做 1 個 KP,之後每批 3–4 個 KP,並必須先詢問老師是否使用多個子代理平行處理(未明說同意就依序)。尚未同步到 course-template 與 pr-2026-fall。

- 2026-10-14 — 老師決定:每個 KP 的簡報頁數不得超過 6 頁(含延伸頁與續頁),agent 必須深度消化來源、依學生程度與簡報時長建構章級與 KP 級知識地圖(老師確認)後,再全新撰寫講義與投影片。技能新增:`NARRATIVE.md` §0(知識地圖與全新撰寫);`LEVELS.md` §7.2–§7.3 改為「細度對應 3–6 頁、放不下的細節放講義」,延伸頁計入 6 頁上限並取消獨立成檔;`audit_deck.py` 新增 `--max-pages-per-kp`(U1,預設 6)與 `--source`(O2 原創性);`audit_lecture.py` 新增 `--source`(L10);`AUDIT_CHECKLIST.md` 新增 S10、S11。尚未同步到 course-template 與 pr-2026-fall。

- 2026-10-14 — 依老師審閱 backprop v2 講義與簡報的意見(頁與頁缺乏邏輯脈絡、延伸頁氾濫、空頁與填充頁、備註套話、Frame 標題被截斷、講義是投影片逐頁轉錄)強化技能:新增 `.agents/skills/NARRATIVE.md`(備課五步、單元敘事弧、銜接條、禁止做法);`view-from-text` 新增 §3E、盤點表的備課筆記與頁序銜接表、Frame 短標題、延伸頁上限與獨立成檔、練習頁規範、備註規範、交付時讀三遍與抽樣自讀;`audit_deck.py` 新增 S1–S4、V1、P2、T2、`--narrative`(R1、R3)、`--outline`;`AUDIT_CHECKLIST.md` 新增 R、S 節;`textbook-lecture-notes` 新增 §4.8 與 `scripts/audit_lecture.py`(L1–L9)。尚未同步到 course-template 與 pr-2026-fall。

- 2026-10-13 — 老師決定:`view-from-text` 加入「簡報時長」。盤點出 KP 數後依 `n × 25` 分鐘建議時長(≤6 個 KP 3 小時、7–9 個 4.5 小時、≥10 個 6 小時),由老師決定後算每 KP 預算並決定細度(完整、標準、精簡);主線放不下的內容做成「延伸:」頁,不刪。`LEVELS.md` 新增 §7;`audit_deck.py` 新增 `--minutes`、`--kps`;`AUDIT_CHECKLIST.md` 新增 T 節。尚未同步到 course-template 與 pr-2026-fall。

- 2026-10-13 — 依老師要求以大學生程度新增 KP02 投影片 `slides/lec03-kp02-gradient-descent-learning-rate-undergrad.jbd.json`(12 個單元、36 頁,中英對照、含講者備註;名詞定義、η=0.05 連續更新表、倍數表與 w² 第三步為補充並已標示)。`view-from-text` 新增規則:不分篇幅,產生簡報前都要先列知識單元表並等老師確認(SKILL §1 第 11 點、AUDIT K1)。與另一份同名主題的 `lec03-kp02-gradient-descent-learning-rate.jbd.json` 並存,檔名以 `-undergrad` 區分。

- 2026-10-13 — 老師決定:講義與投影片技能加入「教材程度」選項(一般大眾、高中生、大學生、研究生、專業人員;預設研究生)。新增共用程度表 `.agents/skills/LEVELS.md`;`textbook-lecture-notes` 與 `view-from-text` 開始前先問程度;`audit_deck.py` 新增 `--level`(檢查每頁公式區塊數與中文正文字數);`AUDIT_CHECKLIST.md` 新增 Q 節;`AGENTS.md` §5.1、§5.11 加提示。尚未同步到 course-template 與 pr-2026-fall。

- 2026-10-13 — KP02／KP03 本機交付驗證完成：主代理重跑內容與結構檢查均無 FAIL，15 張講義附圖全部對應；51 頁近似預覽文字溢出 0，全部 9 張總覽及最後修改的 2 頁完整預覽已人工檢視。保留 KP03 的單一 N3 提醒：原文 0.00001 是高原進展的例示降幅，頁面已標示，並非計算結果。結構檢查只提醒每頁備註會隨下載公開。網站本機建置成功，doctor 無 FAIL。既有 KP01 兩份簡報雜湊未變，未動技能、共用腳本或自動化；尚未提交或同步。

- 2026-10-13 — 依老師確認「好，新增這兩個版本」，由兩個子代理分別以專案 `view-from-text` 製作 KP02「梯度下降與學習率」與 KP03「收斂與停止準則」，主代理整合與人工稽核。新增 `slides/lec03-kp02-gradient-descent-learning-rate.jbd.json`（6 單元、27 內容頁、33 鏡頭）及 `slides/lec03-kp03-convergence-criteria.jbd.json`（4 單元、24 內容頁、28 鏡頭），各有同名講義、附圖與 `assets/figures/src/` 可重現程式。兩者為 16:9、中英對照、自主學習、每頁講者備註，原題全部選項與提示保留，不附選項答案。KP02 的 10 張附圖完整涵蓋原六面板與數值例；無生成資料的原圖保留原始內容，位圖裝入 SVG 容器並非向量重畫。KP03 的 5 張圖區分原 SVG 定性驗證曲線與原公式計算的數值軌跡。澄清原教材 Cauchy 年份、步長與收斂保證條件、一維極大值與鞍點的差別；依原程式核對 Run20：起點 0.02 已達停止條件而更新零次，起點 0.30 在第 11 次更新後停止。第 3 週講義及投影片連結已整合，導覽由既有工具產生。此階段交付本機草稿與近似預覽；講者備註會隨檔案下載，JianBauDau 實際播放、兩種視窗構圖與 PDF 匯出仍待老師確認，尚未提交或同步到雲端。

- 2026-10-13 — 依老師確認，使用專案 `view-from-text` 新增 KP01「最佳化就是參數搜尋」本機草稿：`slides/lec03-kp01-optimization-introduction.jbd.json` 與對應講義 `lectures/lec03-kp01-optimization-introduction.md`。共 3 個知識單元、15 個內容頁（原估 12 頁，4 道檢核題各自拆頁）、18 個鏡頭步驟，中英對照、每頁有講者備註；保留原有 11 頁 KP1 版本。新增 2 張定性曲線圖及可重現程式 `assets/figures/src/lec03-kp01-figures.py`；原教材的三步驟、方法分岔及全章地圖用可編輯節點重建。原文對所有二階方法與平坦度的概括敘述加上適用條件，未加入硬體時間估算。講義及投影片連結已加入第 3 週，網站導覽由既有建置工具產生。結構檢查無 FAIL，內容稽核 FAIL 0、WARN 0，預覽 15 頁、文字溢出 0，網站本機建置成功。預覽是近似渲染；尚待老師在 JianBauDau 確認實際播放、排版及 PDF 匯出。此階段交付本機預覽，尚未同步到雲端。

- 2026-10-13 — 新增 `diagram-design` 技能(`.agents/skills/diagram-design/`,樣式指南已客製為投影片深色色票)與第 3 週 KP1 投影片 `slides/lec03-kp1-parameter-search.jbd.json`(11 頁,中英對照、含講者備註;第 7–10 頁的玩具例子與成本表為補充,已標「[補充]」)。附兩張 diagram-design 圖 `assets/figures/idl-kp1-*`(程式在 `assets/figures/src/`)。老師試用後再決定是否同步到 course-template 與 pr-2026-fall。

- 2026-10-13 — `student-template/` 新增 `AGENTS.md`(給學生的 AI agent 作業助理規範,內容同 `assignments/ai-agent-prompt.md` 並改成 Repo 內使用的版本)、`CLAUDE.md`,README 加一行說明。已推送到 `ndhuccc/idl-2026-hw-template`。

- 2026-10-13 — 老師決定:(1) HW 01 暫時從網站導覽拿掉,等內容定了再放:`assignments/hw01/README.md` 改名為 `README.draft.md`,並在 `mkdocs.yml` 加 `exclude_docs`(還原:改回 `README.md` 並刪除 `exclude_docs` 兩行);(2) 新增給學生的 AI agent 提示詞 `assignments/ai-agent-prompt.md`(建 Repo、習作與 AI 使用紀錄、準備交付文件、繳交、LabTube、口試準備、期末專題),已加到作業說明頁連結與側邊導覽;`scripts/sync_nav.py` 在該檔存在時列入導覽。

- 2026-10-13 — 老師提供 LabTube 網址 http://134.208.2.44:5005;`assignments/README.md`(作業繳交說明頁)加入 LabTube 網址、學生手冊與 PDF 連結,並在繳交流程加上「到 LabTube 登記 Repo 與上傳執行結果」一步。

- 2026-10-13 — 新增 LabTube 學生手冊的課程網站版:`assignments/labtube-student-manual/`(`index.md`、27 張截圖、PDF),內文改成本課程設定(idl-2026-fall、`idl-2026-hw-<學號>`、`ndhuccc/idl-2026-hw-template`),作業說明與老師導覽已加連結。截圖是 pr-2026 示範課程的畫面(課程名稱、`pr-2026-hw-…` 名稱不同),手冊開頭已註明;如要換成 idl 版截圖,需用 LabTube 開發版建立 idl 示範課程重拍。

- 2026-10-13 — 依老師同意建立學生作業範本 Repo `ndhuccc/idl-2026-hw-template`(Public、Template repository,內容為 `student-template/` 目前版本,只含骨架與檢查工具,沒有解答與成績)。網址已寫入 `assignments/README.md`;尚待向學生公告。

- 2026-10-13 — 老師決定:課程簡稱 `idl-2026`(學生作業 Repo 為 `idl-2026-hw-<學號>`);上課地點工A337(每週四下午);助教顏榮興、李哲瑋;Office Hour 週一 10:00–11:00。已寫入 `semester.toml` 並套用,`assignments/README.md`、`hw01`、`SETUP.md`、`student-template/README.md` 的 `<課程簡稱>` 已替換。作業與專題確切截止日待定。

- 2026-10-13 — 老師決定:期末考 15%、期末專題 20%(教學計畫表合計 35%);已寫入 `syllabus.md` 並更新公式。老師要求拿掉先前加的「公開專題 Repo 不得含個資」一條。`scripts/sync_nav.py` 的小修同意同步至範本與 pr-2026-fall;HW 01 內容與日期待內容與日期決定後再做。

- 2026-10-13 — 老師決定並已寫入:(1) 評分照教學計畫表:平時(含出缺席與課堂參與)20%、期中 25%、期末考與期末專題 35%、作業 20%;(2) 刪除範例實驗 `labs/`(Lab 01–03 不計分),連同 `release_schedule.toml`、README、lec01、schedule 的相關連結;(3) 第 17 週(12/31)期末考,期末專題改為學生上傳影片(8–12 分鐘,YouTube 不公開,連結繳交 e 學苑)加 GitHub 報告,不做現場發表;(4) 作業 Repo 維持 Private,期末專題另開獨立 Repo,繳交時才設為 Public;(5) 生成式 AI 使用政策採用課程網頁規則(可用任何 AI 工具,commit 訊息須寫 AI-Tool、AI-Prompt、AI-Reason、Human-Judgment 四欄位,多次小量提交,教師可隨機請學生說明 commit)。來源:老師的課程網頁 https://ndhuccc.github.io/CCC-LNS/IDL_115/Ch01-introduction-history/idl115-w1-dl-intro-history.html。網頁的 HW1–3 只有週次時間窗(第 3→6、8→11、12→15 週),沒有確切日期,已照樣寫入課綱與進度表。另小修 `scripts/sync_nav.py`:沒有對應資料夾的固定導覽項(如已刪除的 labs/)不再列入,避免 mkdocs 警告;此修改尚未同步到範本與 pr-2026-fall。

- 2026-10-13 — 老師決定:上課為每週四;`schedule.md` 依教學計畫表改為 17 週(中英對照主題),第 9 週 11/05 期中考、第 17 週 12/31 期末考/專題發表;第 10 週 11/12 全校運動會停課,RNN(一)改為自主學習(做法 A)。範例實驗 Lab 01–03 保留,排在第 1、3、4 週。首頁與課綱加入教學計畫表的課程目標(中英對照),學習目標依週次主題改寫(老師同意)。

- 2026-10-13 — 依老師指示用最新的 `ndhuccc/course-template` 建立本學期 Repo `ndhuccc/idl-2026-fall`(Private)。原本 10/08 試用建立的同名 Repo 已改名為 `ndhuccc/idl-2026-fall-old` 保留(含範例值,未刪除)。訪談結果:課名「深度學習導論」、課號 CSIE3544AA、2026 秋季、國立東華大學資訊工程學系、江政欽 教授、開學週 2026-09-07;課號、學分、課程描述與教科書取自東華 115/1 教學計畫表(no=115104437)。已套用 `new_semester.py`。

- 2026-10-12 — 同步 `view-from-text` 技能:自讀完整性、計算完整性、符號定義、練習附提示、講義附圖(§4F)、講者備註(§4G)、英文完整句子;`tools/audit_deck.py` 新增 N3、P1、O1、N6 檢查(自 pr-2026-fall 同步通用改進)。

- 2026-10-12 — 新增 `assignments/labtube.md`(LabTube 老師導覽),並在 AGENTS.md、SETUP.md、assignments/README.md 加入指引(自 pr-2026-fall 同步)。

- 2026-10-12 — 同步 `view-from-text` 技能新規則與工具(中英對照、Frame 標準樣式、`tools/audit_deck.py`、`tools/preview_deck.py`),自 pr-2026-fall 同步通用改進。
