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
- [ ] 每週講義與投影片(目前只有範本的 `lec01`)
- [ ] 考試範圍(期中第 9 週、期末第 17 週)

## 決策與異動紀錄

- 2026-10-14 — backpropagation新版教材雲端同步完成：內容版本ab8c011，課程網站發佈工作38455448883成功。線上講義 https://ndhuccc.github.io/idl-2026-fall/lectures/lec03-backprop-optimization-v2/ 、第3週進度及下載頁皆含50頁課堂版與360頁完整版連結；30張講義附圖已逐檔比對。兩份下載檔與已同步版本位元組完全一致，且與本機JSON解析內容相同（Windows換行經標準化）：課堂版SHA-256 79a9a61e0294c59c90052dd777b024a051e47d851c4c58d44f3cc2bfb05ac096；完整版SHA-256 45455b024c5b931437055769ce60710583a4247e8015977f71abb63ebf2fe09a。精準範圍同步70個教材/連結/紀錄檔，網站建置與結構檢查通過；49個既有草稿/人工修改檔雜湊未變，其他本機紀錄與導覽草稿保留未上傳。實際觀感、SVG縮放與PDF匯出仍待老師確認。

- 2026-10-14 — 老師授權「同步雲端」。本次同步backpropagation全章新版講義、50頁課堂版、360頁完整版、對應附圖/繪圖程式及第3週連結；保留其他本機草稿與人工編輯。結構檢查無FAIL、講義公式全對應、360頁近似預覽溢出0，播放器兩種尺寸裁切已修正並重檢。本次內容將由既有課程網站發佈流程更新，線上讀回與下载雜湊尚待確認。

- 2026-10-13 — 依老師指定「大學生、4小時」完成backpropagation第3章KP01–16重製：新增 lectures/lec03-backprop-optimization-v2.md、slides/lec03-backprop-optimization-v2.jbd.json（50主線＋310延伸＝360內容頁，16個Frame、376步）及課堂版 lec03-backprop-optimization-v2-core.jbd.json（50頁、66步）。規劃淨授課200分鐘、休息40分鐘，主線估178分鐘。30張不同講義附圖包含21張數值圖、3張依投影片原生節點/Connector生成的流程圖、6張來源原圖；所有簡報圖片已內嵌，講義圖全部本機可讀。完整保留來源範例、公式與練習選項/提示，明列來源更正及補充，不附練習答案。195個講義公式全對應，427個數學片段嚴格解析錯誤0，結構及內容稽核FAIL 0、360頁近似預覽最終溢出0。播放器1.0.1在1280×720與1920×1080各檢查360頁、392停點及16次返回；發現3頁4個小幅裁切後補高並重檢。測試使用獨立設定、縮短動畫及隱藏視窗定時回呼，實際觀感、SVG縮放與PDF匯出仍待老師確認。保留N3假設輸入0.0008及N6圖中數值/文獻識別碼的提醒並寫明理由；備註會隨下載公開。第3週加入新版講義及兩個簡報連結，網站本機建置成功。保留全部舊版、備份與人工修改，未修改共用腳本/技能/自動化；此階段交付本機預覽，尚未提交或同步到雲端。

- 2026-10-13 — 依老師要求啟動 backpropagation 全章講義與簡報重製。已讀更新後 `view-from-text` 與 `LEVELS.md`，讀取 CCC-LNS 第3章現行主文、16個 KP、教授祕笈、互動數值例與附圖，完成暫存盤點表。依新技能 §1 第10–12點，待老師確認知識單元表、教材程度與簡報時長後才開始批次製作；暫提研究生程度、6小時（淨300分鐘，每KP18.75分鐘，精簡主線），主線估74頁、延伸估200頁。擬新增整章 `lec03-backprop-optimization-v2` 講義及簡報，保留既有版本與人工修改。此階段尚未製作新講義或簡報，也未同步到雲端。

- 2026-10-13 — 依老師要求更新課程網站的 JianBauDau 播放器下載連結與顯示版本為 1.0.1，指向 `ndhuccc/IDL115` 的 `player-v1.0.1` 公開下載檔；同步投影片格式說明。本機網站建置成功，doctor 無 FAIL；透過既有課程網站發佈流程更新。

(最新的在最上面。格式:`YYYY-MM-DD — 決定或異動內容`)

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
