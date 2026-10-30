#!/usr/bin/env python3
"""風格檢查:check_style.py --style diagram|guided <簡報.jbd.json> <證據表.json>
檢查字級下限、頁數與證據表一致、導學版的吉祥物欄位。不判斷畫面美觀;不從畫面偵測吉祥物(只看證據表)。"""
import json, sys

MIN_FONT = 16

def main(a):
    if len(a) != 4 or a[0] != "--style" or a[1] not in ("diagram", "guided"):
        print(__doc__); return 2
    style, deck, ev = a[1], json.load(open(a[2], encoding="utf-8")), json.load(open(a[3], encoding="utf-8"))
    nodes = deck["nodes"]; fails = []
    pages = [k for k, v in nodes.items() if v.get("width") == 1280 and v.get("height") == 720]
    if len(pages) != len(ev): fails.append(f"簡報 {len(pages)} 頁,證據表 {len(ev)} 頁")
    small = [(k, v["style"].get("fontSize")) for k, v in nodes.items()
             if v.get("type") == "text" and (v.get("text") or "").strip() and (v["style"].get("fontSize") or 99) < MIN_FONT]
    if small: fails.append(f"{len(small)} 個文字節點字級 < {MIN_FONT}:{small[:5]}")
    for r in ev:
        m = r.get("mascot")
        if style == "guided" and not (isinstance(m, dict) and m.get("mood") and m.get("prop") and m.get("bubble")):
            fails.append(f"第 {r.get('page')} 頁缺 mascot{{mood,prop,bubble}}")
        if style == "diagram" and m: fails.append(f"第 {r.get('page')} 頁圖解版不應有 mascot")
    for f in fails: print("[FAIL]", f)
    print(f"{style}:{len(pages)} 頁,{'通過' if not fails else '有問題'}")
    return 1 if fails else 0

sys.exit(main(sys.argv[1:]))
