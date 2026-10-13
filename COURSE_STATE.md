# 課程進度與決策紀錄(COURSE_STATE)

> **給 AI agent**:每次工作開始先讀本檔,結束前更新本檔。
> 勾選 `[x]` 表示完成;不要刪除舊紀錄,新紀錄加在「決策與異動紀錄」最上方。
> 此檔會被提交到 Git,**請勿寫入成績、學號、個資、密碼或 Token**。

## 基本資料

| 項目 | 內容 |
|---|---|
| 課程 | (尚未填寫) |
| 學期 | (尚未填寫) |
| Repo | (尚未填寫) |
| Repo 可見度 | (Private / Public — 尚未確認) |
| 使用的 agent 工具 | (例如 Claude Code / Codex / ChatGPT) |
| agent 能力等級 | (L0 / L1 / L2,見 AGENTS.md §1) |

## 建立學期的檢查表(AGENTS.md §3)

- [ ] Step 1 老師有 GitHub 帳號
- [ ] Step 2 用範本建立學期 Repo(老師手動)
- [ ] Step 3 agent 取得 Repo 存取
- [ ] Step 4 完成訪談,產生 `semester.toml`
- [ ] Step 5 套用設定;網站建置與實驗測試驗證通過
- [ ] Step 6 提交並推送
- [ ] Step 7a 啟用 Pages(Source: GitHub Actions)
- [ ] Step 7b 開啟 Discussions
- [ ] Step 7c 手動觸發 `deploy-site` 並確認網站可開
- [ ] Step 8 初始化內容(週次主題、保留的 Lab/作業、解答釋出日、課綱)
- [ ] Step 9 向老師總結

## 待辦事項

(目前沒有)

## 決策與異動紀錄

(最新的在最上面。格式:`YYYY-MM-DD — 決定或異動內容`)

- 2026-10-12 — 同步 `view-from-text` 技能:自讀完整性、計算完整性、符號定義、練習附提示、講義附圖(§4F)、講者備註(§4G)、英文完整句子;`tools/audit_deck.py` 新增 N3、P1、O1、N6 檢查(自 pr-2026-fall 同步通用改進)。

- 2026-10-12 — 新增 `assignments/labtube.md`(LabTube 老師導覽),並在 AGENTS.md、SETUP.md、assignments/README.md 加入指引(自 pr-2026-fall 同步)。

- 2026-10-12 — 同步 `view-from-text` 技能新規則與工具(中英對照、Frame 標準樣式、`tools/audit_deck.py`、`tools/preview_deck.py`),自 pr-2026-fall 同步通用改進。
