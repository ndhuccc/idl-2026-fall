# 教師設定指南(SETUP)

> 本文件給**教師**使用,說明如何把這份範本變成每學期的課程 Repo。
> 標示 🖱️ 的步驟必須在 GitHub 網頁端以你的帳號手動操作。

> 🤖 **想讓 AI Agent 帶著做?** 本文件的流程已寫成給 AI 的操作手冊 `AGENTS.md`;對 GitHub 不熟的老師請看 `START_HERE.md`。

## 一、一次性設定(只需做一次)

1. 🖱️ 在 GitHub 建立 Repo(例如 `course-template`),把本資料夾內容推上去:

   ```bash
   cd course-template
   git init -b main
   git add .
   git commit -m "init: 課程 Repo 範本"
   git remote add origin https://github.com/<你的帳號>/course-template.git
   git push -u origin main
   ```

2. 🖱️ *Settings → General → Template repository* 打勾,使其成為範本。
3. (建議)驗證範本本身健康:

   ```bash
   pip install -r requirements.txt
   LAB_TARGET=solution pytest labs -q   # 解答應全數通過
   pytest labs -q                        # starter 應全數失敗(表示測試有鑑別力)
   ```

## 二、每學期流程

### 步驟 1:由範本建立新 Repo
🖱️ 範本頁面按 **Use this template → Create a new repository**,命名如 `dl-2026-fall`。

### 步驟 2:填寫學期設定並套用

```bash
git clone https://github.com/<你的帳號>/dl-2026-fall.git && cd dl-2026-fall
cp semester.example.toml semester.toml     # 編輯 semester.toml
python scripts/new_semester.py --dry-run   # 預覽會改哪些檔案
python scripts/new_semester.py             # 正式套用(需要 Python 3.11+)
```

腳本會替換所有 `{{佔位符}}`(課名、學期、聯絡方式、`W01`~`W18` 週次日期等),
並列出**尚未填入**的佔位符,讓你補齊。

> 📌 **`deploy-site` 在範本 Repo 不會執行**(範本的 `mkdocs.yml` 還是佔位符,建置會失敗)。
> 要測試網站發佈,請在**套用過 `new_semester.py` 的學期 Repo** 操作:
> 該 Repo → *Actions* → 左側選 **deploy-site** → 右側 **Run workflow** → 選 `main` → 綠色 *Run workflow*。
> 需先完成下方的 Pages 設定(Source: GitHub Actions)。
> 私有 Repo 使用 Pages 需要 GitHub Pro / Team 等付費方案;免費帳號請改用公開 Repo(此時請勿放 `solution/`)。

### 步驟 3:啟用 GitHub 功能(🖱️ 網頁端)

| 功能 | 位置 | 說明 |
|---|---|---|
| Pages | Settings → Pages → Source 選 **GitHub Actions** | 課程網站;網址即 `SITE_URL` |
| Discussions | Settings → General → Features | 開放討論區(**新 Repo 預設關閉,不會從範本繼承,每學期都要重新勾選**) |
| Issues | 預設已啟用 | 已提供提問/勘誤/申請範本 |
| Projects | Projects → New project | 教學進度與專題里程碑看板 |
| Codespaces | 預設可用 | 已含 `.devcontainer/` |

### 步驟 4:作業繳交與評分(學生自建 Repo,無需設定)

> LabTube 作業系統的開學前設定與每份作業流程,見 `assignments/labtube.md`。

本課程作業**只採用一種流程**:每位學生建立**一個**私人 Repo(`idl-2026-hw-<學號>`)存放整學期作業,並把教師帳號 `ndhuccc` 加為協作者;
每次作業在 `hwNN/` 資料夾完成,截止前打標籤 `hwNN-submit` 即完成繳交,不需另外寄網址。完整規範見 `assignments/README.md`。

教師端不需要設定 Classroom,只需接受學生的協作邀請。評分時請 AI agent 依 `.agents/skills/assignment-grading/SKILL.md` 讀取各 Repo,
只依 `hwNN/README.md` 的 Rubric 初評,**成績不寫進任何 Repo**,最終由老師定案。

**建立學生作業 Repo 範本(一次性,🖱️)**

`student-template/` 是學生作業 Repo 的範本內容(根目錄 `README.md`、`.gitignore`、`hw01/` 骨架、`tools/check_structure.py`、
`tools/new_hw.py`、`.github/workflows/check-structure.yml`)。請把它做成獨立的 GitHub 範本 Repo,讓學生一鍵建立:

1. 在教師帳號下建立一個新的空 Repo,名稱為 `idl-2026-hw-template`。範本內**沒有解答與成績**,可設為 Public,學生才能使用。
2. 把 `student-template/` 的**內容**(不含資料夾本身)放到該 Repo 根目錄並推送。
3. 到該 Repo 的 **Settings → General**,勾選 **Template repository**。
4. 把範本網址公告給學生(課程網站或課堂),並在 `assignments/README.md` 補上網址。
5. 之後修改 `student-template/` 時,同步到該範本 Repo(學生已建立的 Repo 不會自動更新)。

> 提醒:
> - 本課程的「課程簡稱」是 `idl-2026`:學生作業 Repo 名稱為 `idl-2026-hw-<學號>`,LabTube 本課程的「學生作業 Repo 名稱前綴」也要設為 `idl-2026`(學生登記的 Repo 名稱必須是「簡稱-hw-學號」)。
> - 每份新作業的 `hwNN/README.md` 都要寫清楚繳交內容、截止時間與 Rubric;Rubric 是 AI 評分唯一的依據。
> - 作業 Repo 由學生建立,不會取得本 Repo 的 `labs/*/solution/`,但仍請遵守下節的解答保密規則。

### 步驟 5:設定解答釋出日

編輯 `release_schedule.toml`,為每個實驗填入日期。`deploy-site.yml` 每天自動重建網站,
日期一到,`labs/<lab>/solution.md` 就會出現在網站上,**不需手動操作**。

## 三、解答保密(重要)

`release_schedule.toml` 只控制**網站**。只要 `solution/` 與公開 Repo 一起被推送,
任何人都能在 Repo 內看到它。依你的需求擇一:

| 做法 | 適用 | 作法 |
|---|---|---|
| **A. 教師私有 Repo(推薦)** | 一般情況 | 維護一個私有 Repo 含完整內容(含 solution);只把「去除 solution」的版本公開或發給學生 |
| B. Repo 全程私有 | 修課人數少、Repo 只給修課學生 | 將學生加為 Collaborator(唯讀);網站另行發佈(私有 Repo 的 Pages 需付費方案) |
| C. 公開 Repo + 不放解答 | 不需自動釋出解答 | 解答另存他處,於截止後手動貼到網站 |

## 四、日常維運

| 想做的事 | 做法 |
|---|---|
| 新增講義 | 在 `lectures/` 新增 `lecNN.md`,推送後網站自動更新 |
| 新增實驗 | `cp -r labs/lab01 labs/lab02`,修改後在 `release_schedule.toml` 加日期 |
| 公告異動 | 編輯 `CHANGELOG.md`,並視需要發 Discussions 公告 |
| 學期結束封存 | 🖱️ Releases → 建立 `v{{學期}}` Tag;Settings → 勾選 Archive this repository |

## 五、佔位符一覽

| 佔位符 | 來源 |
|---|---|
| `COURSE_NAME` `COURSE_CODE` `SEMESTER` `UNIVERSITY` `DEPARTMENT` `INSTRUCTOR` `EMAIL` `GITHUB_USER` `REPO_NAME` `START_DATE` | `semester.toml` 必填 |
| `COURSE_DESCRIPTION` `CREDITS` `CLASS_TIME_LOCATION` `TA` `OFFICE_HOUR` `TEXTBOOK` `AI_POLICY` `TEAM_SIZE` | `semester.toml` 選填 |
| `REPO_URL` `SITE_URL` `YEAR` `W01`–`W18` | 自動推導 |
| 其他(如 `HW01_DUE`) | `semester.toml` 的 `[extra]` 區段 |

## 六、隱私與合規提醒

- **成績、學號、學生個資絕不放入公開 Repo**;`.gitignore` 已排除 `grades/` 與 `private/`。
- 使用學生程式碼相似度比對工具前,請先在課綱中告知學生。
- 公開考古題前,確認不會與本學期考題重複。
