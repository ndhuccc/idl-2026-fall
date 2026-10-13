# 課程專用 Repo 逐步實現規劃書

> 目標:每學期初,以本範本一鍵建立「課程專用 Repo」,內含經營課程所需的**資源**與**服務**。

## 一、總體策略

| 原則 | 做法 |
|---|---|
| 可複製 | 本 Repo 設為 GitHub **Template Repository**;`scripts/new_semester.py` 一鍵替換學期參數 |
| 單一入口 | `README.md` + GitHub Pages 課程網站 |
| 公私分離 | 公開:講義、作業說明;私有:解答、成績(由 Actions 排程釋出解答) |
| 環境一致 | `requirements.txt` + `.devcontainer/`(Codespaces) |
| 自動化優先 | 連結檢查、Notebook 驗證、作業自動評分、網站發佈 |

## 二、階段規劃

### 階段 0:規劃(本文件)
- **交付物**:`PLAN.md`
- **驗收**:階段、交付物、驗收標準皆明確

### 階段 1:課程基本文件
- **交付物**:`README.md`、`syllabus.md`、`schedule.md`、`CHANGELOG.md`、`FAQ.md`、`glossary.md`、`CONTRIBUTING.md`、`.gitignore`、`LICENSE`
- **驗收**:所有文件含 `{{佔位符}}`,可被 `new_semester.py` 替換

### 階段 2:教學與實作內容骨架
- **交付物**:`lectures/`、`slides/`、`notebooks/`、`labs/lab01/{starter,solution,tests}`、`assignments/hw01`、`projects/`、`exams/`、`quizzes/`、`references/`、`assets/`、`study-guide/`
- **驗收**:每個目錄有 README 說明用途;`lab01` 的測試可對 solution 通過、對 starter 失敗(證明評分有鑑別力)

### 階段 3:GitHub 協作與自動化
- **交付物**:
  - Issue 範本:提問、勘誤、請假/申請
  - PR 範本
  - Actions:`link-check`、`notebook-check`、`autograde`、`deploy-site`
  - 解答釋出:不另設工作流程,改由 `release_schedule.toml` + `scripts/build_docs.py` 判斷,`deploy-site` 每日重建即自動上線
- **驗收**:YAML 語法皆可被解析;工作流程觸發條件與權限最小化

### 階段 4:課程網站、環境與學期複製
- **交付物**:`mkdocs.yml`(含 LaTeX)、`requirements.txt`、`.devcontainer/devcontainer.json`、`scripts/new_semester.py`、`SETUP.md`
- **驗收**:`mkdocs build` 成功;`new_semester.py` 能正確替換參數

### 階段 5:驗證與交付
- **驗收**:`pytest` 通過、YAML/JSON 解析通過、網站建置成功、打包為 `.zip`

## 三、風險與對策

| 風險 | 對策 |
|---|---|
| 解答外洩 | 網站只在 `release_schedule.toml` 日期後才含解答;**但公開 Repo 內的 `solution/` 仍可見**,保密需用教師私有 Repo(見 `SETUP.md`) |
| 學生個資/成績外洩 | 範本不含成績;`.gitignore` 排除 `grades/` |
| Actions 額度 | 僅在必要路徑變更時觸發(`paths:` 過濾) |
| 平台限制 | Template Repository、Pages、Classroom 需在 GitHub 網頁端手動啟用(見 `SETUP.md`) |

## 四、使用者需手動完成的步驟

以下需在你的 GitHub 帳號操作,我無法代為執行:
1. 建立 Repo 並推送本範本
2. 勾選 *Settings → Template repository*
3. 啟用 *Settings → Pages*(Source: GitHub Actions)
4. 如需自動批改,於 GitHub Classroom 建立作業

## 五、進度追蹤

- [x] 階段 0 規劃
- [x] 階段 1 基本文件
- [x] 階段 2 內容骨架(Lab 01:starter 8 項全失敗、solution 8 項全通過)
- [x] 階段 3 協作與自動化
- [x] 階段 4 網站與環境
- [x] 階段 5 驗證交付(YAML/JSON/TOML 解析、new_semester、解答釋出日、mkdocs build)

## 六、後續擴充紀錄

- [x] Lab 02:NumPy 手寫反向傳播 + 數值梯度檢查(starter 11 項全失敗、solution 全通過)
- [x] Lab 03:PyTorch MLP 與訓練流程(需 torch;未安裝時 skip)
- [x] `autograde` 的 `grade` 工作改為只在非範本 Repo 執行
- [x] Agent 化:`AGENTS.md`(操作手冊)、`CLAUDE.md`、`START_HERE.md`(老師入口)、`COURSE_STATE.md`(跨對話進度)、`scripts/doctor.py`(現況自我檢查)
- [x] 透明化:AGENTS.md 加入「對老師隱藏 Git/GitHub」術語對照,以及 gh 全自動路徑(Step 2-A、7-A);START_HERE.md 依 AI 工具類型改寫;doctor.py 回報 gh 登入狀態(**gh 全自動路徑尚未在真實帳號上實測**,本環境不允許建立 Repo/修改設定)
