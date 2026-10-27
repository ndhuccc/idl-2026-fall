# view-from-text-v3

已核准講義 → 頁面規劃 → HTML/CSS/SVG（KaTeX）→ JianBauDau 原生節點 → 稽核 → 教師確認 → 本課發布。

完整流程見 SKILL.md，視覺規則見 DESIGN.md。

從課程根目錄建立單元：

    python3 .agents/skills/view-from-text-v3/tools/new_unit.py /tmp/unit-work KP03 "單元名" "Unit name" --lecture lec07
    MIN=40 bash .agents/skills/view-from-text-v3/tools/run_unit.sh /tmp/unit-work kp03_deck

預設跟隨工具所在課程；可用 --repo 或 COURSE_REPO 明確指定。課名從本課設定讀取，章節必須指定，course-unit.json 保存單元設定。空白腳手架須先填入核准的頁面內容再建置。

發布預覽用 publish_unit.py --dry-run，只列計畫，不改檔。跨課程同步須明確加 --sync-to，日誌須明確加 --log-file；詳見 SKILL.md。examples/ 保留歷史 PR 教材供設計參考，不作新課程內容的預設。

驗證：

    python3 -m unittest discover -s .agents/skills/view-from-text-v3/tests -v
