# image-to-jbd-page

把一張圖片精準復現成 JianBauDau 原生簡報頁。入口是 [SKILL.md](SKILL.md)。

快速試用(需要 Pillow、Playwright + Chromium):

    python3 examples/example_page.py /tmp/demo
    python3 tools/preview_deck.py /tmp/demo/demo.jbd.json --out /tmp/demo/pv

整合到其他 agent:把整個資料夾放到該 agent 的技能目錄(例如 `.agents/skills/`、`~/.claude/skills/`),讓它讀 SKILL.md 即可;所有工具只依賴本資料夾內的檔案。
