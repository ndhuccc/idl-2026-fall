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

(最新的在最上面。格式:`YYYY-MM-DD — 決定或異動內容`)

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
