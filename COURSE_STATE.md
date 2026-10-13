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
- [ ] Step 7a 啟用 Pages(Source: GitHub Actions)
- [ ] Step 7b 開啟 Discussions
- [ ] Step 7c 手動觸發 `deploy-site` 並確認網站可開
- [ ] Step 8 初始化內容(週次主題、保留的 Lab/作業、解答釋出日、課綱)
- [ ] Step 9 向老師總結

## 待辦事項

- [ ] 選填欄位:上課時間地點、助教、Office Hour、專題組隊人數、生成式 AI 使用政策(老師決定)、HW01 發佈與截止日
- [ ] Step 7:啟用課程網站與討論區、第一次發佈
- [ ] Step 8:依教學計畫表填 `schedule.md` 週次主題(17 週,第 9 週期中、第 17 週期末/專題)與 `syllabus.md` 評分(平時 20%、期中 25%、期末 35%、作業 20%);決定範例 Lab 去留與解答釋出日
- [ ] 作業與 LabTube:決定課程簡稱、建立學生作業範本 Repo、在 LabTube 建課並設定作業 Repo 前綴、放學生手冊

## 決策與異動紀錄

(最新的在最上面。格式:`YYYY-MM-DD — 決定或異動內容`)

- 2026-10-13 — 依老師指示用最新的 `ndhuccc/course-template` 建立本學期 Repo `ndhuccc/idl-2026-fall`(Private)。原本 10/08 試用建立的同名 Repo 已改名為 `ndhuccc/idl-2026-fall-old` 保留(含範例值,未刪除)。訪談結果:課名「深度學習導論」、課號 CSIE3544AA、2026 秋季、國立東華大學資訊工程學系、江政欽 教授、開學週 2026-09-07;課號、學分、課程描述與教科書取自東華 115/1 教學計畫表(no=115104437)。已套用 `new_semester.py`。

- 2026-10-12 — 同步 `view-from-text` 技能:自讀完整性、計算完整性、符號定義、練習附提示、講義附圖(§4F)、講者備註(§4G)、英文完整句子;`tools/audit_deck.py` 新增 N3、P1、O1、N6 檢查(自 pr-2026-fall 同步通用改進)。

- 2026-10-12 — 新增 `assignments/labtube.md`(LabTube 老師導覽),並在 AGENTS.md、SETUP.md、assignments/README.md 加入指引(自 pr-2026-fall 同步)。

- 2026-10-12 — 同步 `view-from-text` 技能新規則與工具(中英對照、Frame 標準樣式、`tools/audit_deck.py`、`tools/preview_deck.py`),自 pr-2026-fall 同步通用改進。
