#!/usr/bin/env python3
"""plan_pages.py -- plan the pages of a knowledge point FIRST, so figures that cover several lecture sections are chosen at inventory
time (one pass), not merged afterwards.

    python3 plan_pages.py plan.json lecture.md [--modality-out modality.json]

plan.json:
  {"kp": "KP23", "pages": [
     {"title": "...", "covers": ["3. 不純度"], "main": "圖|表|算式|文字|題目", "layout": "columns_compare|two_options|before_after|steps_flow|freeform|none",
      "reason": "only for main != 圖: 讀者要..., 圖幫不上因為..."}, ...]}
`covers` entries match lecture sections (lines starting with ###) by substring. Checks:
  P1 every lecture section is covered by at least one page            (FAIL)
  P2 main is one of the five kinds; layout is registered or freeform  (FAIL)
  P3 a page that is not 圖 needs a reason that mentions 讀者           (FAIL)
  P4 a page covering 2+ sections should be 圖 with a layout           (WARN: a table or cards page is probably two pages in disguise)
  P5 more pages than lecture sections                                  (WARN: plan figures that cover several sections)
  P6 the same section is the only content of 3+ pages                  (WARN)
  P7 a figure page lacks its evidence-table fields: question (one core question), takeaway (one sentence the reader must keep),
     numbers (what is shown and how it was recomputed: "recomputed from unrounded values", "textbook prints X, recomputed Y", or "none"),
     source (lecture section / page it comes from)        (WARN; fields come from page-to-teaching-diagram, step 2 "content evidence table")
Prints the coverage matrix; writes modality.json for audit_deck.py --modality (page number -> kind, reason, layout).
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
KINDS = ("圖", "表", "算式", "文字", "題目")


def sections(md):
    return [re.sub(r"^#+\s*", "", ln).strip() for ln in md.splitlines() if ln.startswith("### ")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan"); ap.add_argument("lecture"); ap.add_argument("--modality-out")
    a = ap.parse_args()
    plan = json.load(open(a.plan, encoding="utf-8")); secs = sections(open(a.lecture, encoding="utf-8").read())
    try:
        from figkit import layouts
        known = set(layouts.names()) | {"freeform", "none"}
    except Exception:
        known = {"columns_compare", "two_options", "before_after", "steps_flow", "freeform", "none"}
    fails, warns = [], []
    cover = {s: [] for s in secs}
    for i, p in enumerate(plan["pages"], 1):
        if p["main"] not in KINDS: fails.append(f"P2 page {i} '{p['title']}': main must be one of {KINDS}")
        if p.get("layout", "none") not in known: fails.append(f"P2 page {i}: unknown layout {p.get('layout')!r} (known: {sorted(known)})")
        if p["main"] == "圖" and p.get("layout", "none") == "none": fails.append(f"P2 page {i}: main 圖 needs a layout (or freeform)")
        if p["main"] != "圖" and ("讀者" not in p.get("reason", "") or len(p.get("reason", "")) < 10):
            fails.append(f"P3 page {i} '{p['title']}': not a figure page: reason must say what the 讀者 does and why a figure would not help")
        if p["main"] == "圖":
            miss = [k for k in ("question", "takeaway", "numbers", "source") if not p.get(k)]
            if miss: warns.append(f"P7 page {i} '{p['title']}': evidence-table fields missing: {', '.join(miss)}")
        hit = []
        for c in p.get("covers", []):
            m = [s for s in secs if c in s]
            if not m: warns.append(f"page {i}: covers {c!r} matches no lecture section")
            for s in m: cover[s].append(i); hit.append(s)
        if len(set(hit)) >= 2 and not (p["main"] == "圖" and p.get("layout", "none") not in ("none",)):
            warns.append(f"P4 page {i} '{p['title']}' covers {len(set(hit))} sections but is not a layout figure: probably several pages in disguise")
    for s, pg in cover.items():
        if not pg: fails.append(f"P1 lecture section '{s}' is not covered by any page")
        if len(pg) >= 3: warns.append(f"P6 section '{s}' fills pages {pg}: merge them into one figure")
    if len(plan["pages"]) > len(secs): warns.append(f"P5 {len(plan['pages'])} pages for {len(secs)} lecture sections: plan figures that cover several sections")
    print(f"{plan.get('kp', '')}: {len(plan['pages'])} pages, {len(secs)} lecture sections")
    for s, pg in cover.items(): print(f"  {s[:34]:36s} -> pages {pg}")
    for w in warns: print("  WARN", w)
    for f in fails: print("  FAIL", f)
    print(f"FAIL {len(fails)}  WARN {len(warns)}")
    if a.modality_out:
        json.dump({str(i): {"kind": p["main"], "reason": p.get("reason", "") or f"layout {p.get('layout')}", "layout": p.get("layout", "none")} for i, p in enumerate(plan["pages"], 1)},
                  open(a.modality_out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
