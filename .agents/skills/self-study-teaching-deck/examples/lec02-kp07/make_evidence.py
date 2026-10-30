#!/usr/bin/env python3
"""由 kp07_shared.py 產生兩份證據表:evidence-diagram.json(無吉祥物)與 evidence-guided.json(含吉祥物欄位)。"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import kp07_shared as SH
MATH = SH.specs.MATH
base, guided = [], []
for i, p in enumerate(SH.S, 1):
    kind, tz, te, cz, ce, sec, vis, prop, mood, bz, be = p
    r = dict(page=i, title=tz, takeaway=cz, source_anchors=SH.ANCHORS[i - 1], visual=vis, math=MATH.get(kind, []))
    if i > 1: r["bridge"] = SH.BR[i - 1][0]
    if i in SH.NOTE: r["note"] = SH.NOTE[i]
    if kind.startswith(("practice", "quiz", "myths")): r["self_check"] = True
    if kind.startswith("practice") and i in (24, 25): r["note"] = "參考答案為補充(講義只給提示),數值由程式驗算。"
    if kind.startswith("quiz"): r["note"] = "參考答案依講義內容推得;講義原文不附答案,老師可決定是否保留。"
    base.append(r)
    guided.append(dict(r, mascot=dict(mood=mood, prop=prop, bubble=bz.replace("\n", ""))))
json.dump(base, open(os.path.join(HERE, "evidence-diagram.json"), "w"), ensure_ascii=False, indent=1)
json.dump(guided, open(os.path.join(HERE, "evidence-guided.json"), "w"), ensure_ascii=False, indent=1)
print("pages", len(base))
