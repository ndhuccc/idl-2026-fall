#!/usr/bin/env python3
"""講義覆蓋完整度檢查。
  --list <講義.md>            列出講義錨點(證據表的 source_anchors 要用這些字串)
  <講義.md> <證據表.json>     檢查:每個錨點至少對到一頁、每頁都有來源、自學欄位齊全
有問題以非 0 結束。"""
import json, re, sys

CJK = re.compile(r"[一-鿿]")


def short(s):
    s = re.sub(r"[（(].*$", "", s).strip()
    return s.strip("*：: ")


def anchors(md):
    out, sec, label, n = [], "", "", 0
    for line in md.splitlines():
        m = re.match(r"^(#{2,3})\s+(.*)", line)
        if m:
            sec = short(m.group(2)); label = ""; n = 0
            if m.group(1) == "###" and not sec.startswith(("術語對照", "學習目標", "符號表")) or sec.startswith(("學習目標", "符號表")):
                out.append(sec)
            continue
        m = re.match(r"^\*\*([^*]+)\*\*", line)
        if m and CJK.search(m.group(1)):
            label = short(m.group(1)); n = 0
            if sec.startswith("本單元重點整理") and label not in ("常見迷思", "小練習", "解題示範", "自我檢核", "想深入"):
                out.append(f"重點:{label}")
            continue
        if sec.startswith("本單元重點整理") and label in ("常見迷思", "小練習", "解題示範", "自我檢核", "想深入"):
            m = re.match(r"^(?:- |(\d+)\. |\|\s*(\d+)\.)", line)
            if m and label in ("常見迷思", "小練習", "自我檢核") and CJK.search(line):
                n += 1; out.append(f"{label}#{n}")
            elif label == "解題示範" and re.match(r"^1\. ", line):
                out.append("解題示範")
            elif label == "想深入" and line.strip() and f"{label}" not in out:
                out.append(label)
    seen, res = set(), []
    for a in out:
        if a not in seen:
            seen.add(a); res.append(a)
    return res


def main(a):
    if len(a) == 2 and a[0] == "--list":
        print("\n".join(anchors(open(a[1], encoding="utf-8").read()))); return 0
    if len(a) != 2:
        print(__doc__); return 2
    anc = anchors(open(a[0], encoding="utf-8").read())
    rows = json.load(open(a[1], encoding="utf-8"))
    bad = []; cover = {x: [] for x in anc}
    for r in rows:
        p = r.get("page")
        for k in ("title", "takeaway", "source_anchors", "visual", "math"):
            if k not in r: bad.append(f"第 {p} 頁缺欄位 {k}")
        if p and p > 1 and not r.get("bridge"): bad.append(f"第 {p} 頁缺 bridge(承接)")
        src = r.get("source_anchors") or []
        if not src: bad.append(f"第 {p} 頁沒有來源錨點")
        for s in src:
            if s == "(新增)":
                if not r.get("note"): bad.append(f"第 {p} 頁是新增內容但沒有 note 說明")
            elif s in cover: cover[s].append(p)
            else: bad.append(f"第 {p} 頁的錨點「{s}」不在講義錨點清單")
    miss = [x for x, v in cover.items() if not v]
    print(f"頁數 {len(rows)};講義錨點 {len(anc)};已覆蓋 {len(anc)-len(miss)}")
    for x in miss: print(f"[FAIL] 錨點未覆蓋:{x}")
    for b in bad: print(f"[FAIL] {b}")
    if not any(r.get("self_check") for r in rows): print("[WARN] 沒有任何自我檢查頁(self_check)")
    return 1 if miss or bad else 0

sys.exit(main(sys.argv[1:]))
