#!/usr/bin/env python3
"""Recompute every chain of purely numeric equalities printed on the slides ($1+1-0-1.5=0.5$, ...).

usage: check_arithmetic.py deck.jbd.json [...]   exit 1 if any printed chain is not equal.
It checks what the page PRINTS, not what the author's own program computes, so a wrong or
mismatched printed formula is caught (AUDIT_CHECKLIST V)."""
import json, re, sys

NUM = re.compile(r"^[\d\s.+\-*/()]+$")

def clean(seg):
    s = seg.strip().replace("−", "-").replace("\\,", "").replace("\\ ", "").replace("\\cdot", "*").replace("\\times", "*")
    s = re.sub(r"\\tfrac\{(\d+)\}\{(\d+)\}|\\frac\{(\d+)\}\{(\d+)\}", lambda m: f"({m[1] or m[3]}/{m[2] or m[4]})", s)
    s = re.sub(r"\\tfrac(\d)(\d)", r"(\1/\2)", s)
    s = s.replace("{", "(").replace("}", ")")
    return s

def chains(text):
    for m in re.finditer(r"\$([^$]+)\$", text):
        parts = [clean(p) for p in m.group(1).split("=")]
        nums = []
        for p in parts:
            if p and NUM.match(p) and re.search(r"\d", p):
                try: nums.append((p, eval(p, {"__builtins__": {}})))
                except Exception: pass
        if len(nums) >= 2: yield m.group(0), nums

def main(paths):
    bad = 0
    for path in paths:
        doc = json.load(open(path, encoding="utf-8"))
        n_ok = 0; n_round = 0
        for nid, n in doc["nodes"].items():
            if n["type"] not in ("text", "shape") or not isinstance(n.get("text"), str): continue
            for src, nums in chains(n["text"]):
                spread = max(abs(v - nums[0][1]) for _, v in nums)
                scale = max(abs(v) for _, v in nums) or 1.0
                if spread > 1e-9 and spread / scale > 0.005:
                    bad += 1; print(f"MISMATCH {path.split('/')[-1]} {nid}: {src}  ->  {[round(v, 6) for _, v in nums]}")
                else:
                    n_ok += 1          # exact, or only a printed-rounding difference (< 0.5 %)
                    if spread > 1e-9: n_round += 1
        print(f"{path.split('/')[-1]}: {n_ok} numeric chains verified ({n_round} differ only by printed rounding)")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
