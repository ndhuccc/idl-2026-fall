# view-from-text-v3

講義 → JianBauDau 原生簡報:先規劃,再用 HTML/CSS/SVG(KaTeX)自由設計圖,自動轉成原生節點。流程與規則見 `SKILL.md`,設計限制見 `DESIGN.md`。

快速開始:
```bash
python3 tools/new_unit.py /tmp/kp17 KP17 "單元名" "Unit name"
python3 examples/kp18_pnn.py /tmp/out       # 範例:HTML 設計 → zh/en 的 .html/.png/.json
MIN=40 bash tools/run_unit.sh /tmp/kp17 kp17_deck
```
