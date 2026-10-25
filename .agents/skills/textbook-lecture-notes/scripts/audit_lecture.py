#!/usr/bin/env python3
"""Automatic quality audit for a bilingual lecture note (lectures/lecNN*.md).

Checks what a program can check (SKILL.md section 7, NARRATIVE.md):
  L1  every heading has an English part in full-width parentheses, no placeholder English ("Knowledge point 09")
  L2  headings that repeat three or more times (boilerplate sections)
  L3  slide / class-time vocabulary that does not belong in a lecture (main line, extension pages, minutes, core allocation)
  L4  learning objectives that are the same sentence for several knowledge points
  L5  in-page links (#anchor) that do not match any heading
  L6  numbers written as Chinese numerals in headings (hard to search)
  L7  each knowledge point contains the SKILL.md 4.6 fields (one-sentence summary, plain explanation, formal definition,
      example, common misconception, practice)
  L8  slide-by-slide transcription: too many tiny sections per knowledge point
  L9  slides / sections that are only a one-line fragment (< 40 characters of body)
  L11 display math ($$ ... $$): a closing $$ line directly followed by an opening $$ line (no blank line between),
      or an odd number of $$ lines. The site renders the stray "$$" as visible text ("$$$$" glitch). FAIL.
Added 2026-10-26 (readability for weak-background readers, see CLARITY.md C19 and the lec05 audits):
  L12 sentence length: zh sentences over 40 characters / en sentences over 25 words (math counted as one token)
  L13 paragraphs of more than 4 sentences (FAIL when more than 40% of the paragraphs)
  L14 no symbol table although the lecture uses many formulas
  L15 glossary coverage: bold-defined terms and acronyms that are missing from the glossary table
  L16 practice questions without a worked solution (FAIL)
  L17 overlap with the slide deck: the lecture must not be a transcription of the deck (WARN above 20%, FAIL above 50%)
  L18 figure captions without a reading guide / function mark (〔功能：…〕 or [Function: …])
  L19 performance numbers (accuracy, recall, error ...) without any setup word (seed, runs, test set) in the lecture
  L20 skipped-step wording ("顯然", "易知", "不難看出", "obviously" ...)
  L21 teaching text too short (under 1200 zh characters / 700 en words, math and captions excluded)
  L22 a defined term is used before its definition (headings, objectives and the one-sentence summary are exempt)
Usage:  py .agents/skills/textbook-lecture-notes/scripts/audit_lecture.py lectures/lecNN.md
Exit code 1 when there is a FAIL.
"""
import argparse
import re
import sys
from collections import Counter

HEAD = re.compile(r"^(#{2,3})\s+(.*)$", re.M)
FIELDS = [("一句話重點", "One-sentence"), ("白話說明", "Plain explanation"), ("正式定義", "Formal definition"),
          ("例子", "Example"), ("常見迷思", "Common misconception"), ("小練習", "Practice")]
META = re.compile(r"主線頁|延伸頁|課堂主線|Core allocation|簡報時長|淨授課|休息|core minutes|主線時間")
PLACEHOLDER = re.compile(r"Knowledge point\s*\d+|knowledge-point|Section\s*\d+\s*[)）]", re.I)
CN_NUM = re.compile(r"[一二三四五六七八九十百千萬兆]+[點兆萬億]|[點][一二三四五六七八九零]")


def slug(h):
    h = re.sub(r"<[^>]+>", "", h).strip().lower()
    h = re.sub(r"[^\w\s-]", "", h, flags=re.U)
    return re.sub(r"[\s]+", "-", h)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lecture")
    ap.add_argument("--source", action="append", default=[], help="source text file(s); L10 warns when the lecture repeats long runs of the source (14+ identical CJK characters)")
    ap.add_argument("--lang", choices=["bi", "zh", "en"], default="bi", help="bi (default): bilingual lecture; zh / en: single-language version (skips the English-heading check L1; en looks for the English field names in L7)")
    ap.add_argument("--deck", help="slide deck (.jbd.json) for L17; default: slides/<lecture stem>[-en].jbd.json when it exists")
    ap.add_argument("--max-sections-per-kp", type=int, default=14, help="L8 limit (default 14)")
    args = ap.parse_args()
    text = open(args.lecture, encoding="utf-8").read()
    fails, warns = [], []
    F = lambda c, m: fails.append(f"FAIL {c} {m}")
    W = lambda c, m: warns.append(f"WARN {c} {m}")

    heads = [(len(m.group(1)), m.group(2).strip(), m.start()) for m in HEAD.finditer(text)]
    # L1
    for lvl, h, _ in heads:
        if PLACEHOLDER.search(h): F("L1", f"placeholder English in heading: {h!r}")
        elif args.lang == "bi" and ("（" not in h and "(" not in h) and not re.match(r"^(符號表|術語對照|學習目標)", h) and re.search(r"[一-鿿]", h):
            W("L1", f"heading has no English part: {h!r}")
    # L2
    zh = Counter(re.split(r"[（(]", h)[0].strip() for lvl, h, _ in heads if lvl == 3)
    for t, c in zh.items():
        if c >= 3 and not re.match(r"^(符號表|術語對照|術語表|學習目標|來源)", t): W("L2", f"heading {t!r} appears {c} times (boilerplate sections)")
    # L3
    for m in META.finditer(text):
        ln = text.count("\n", 0, m.start()) + 1
        W("L3", f"line {ln}: slide/class-time wording in a lecture: {m.group(0)!r}")
        break
    n_meta = len(META.findall(text))
    if n_meta > 3: F("L3", f"{n_meta} places use slide/class-time wording; the lecture must not depend on the deck or the class schedule")
    # L4
    objs = re.findall(r"學習目標[^\n]*\n+([^\n]+)", text)
    norm = Counter(re.sub(r"[「『][^」』]*[」』]", "X", o) for o in objs)
    for o, c in norm.items():
        if c >= 3: W("L4", f"{c} learning objectives are the same sentence ({o[:30]!r}...): write objectives specific to each knowledge point")
    # L5
    anchors = {slug(h) for _, h, _ in heads}
    for m in re.finditer(r"\]\(#([^)]+)\)", text):
        if m.group(1).lower() not in anchors and m.group(1) not in anchors:
            W("L5", f"link #{m.group(1)} matches no heading")
    # L6
    for lvl, h, _ in heads:
        if CN_NUM.search(h): W("L6", f"Chinese numerals in a heading (use digits so it can be searched): {h!r}")
    # L11 display-math fences: $$ lines must pair up, and two blocks need a blank line between them
    lines = text.splitlines()
    fence = [i for i, l in enumerate(lines) if l.strip() == "$$"]
    if len(fence) % 2:
        F("L11", f"odd number of '$$' lines ({len(fence)}): an unclosed display equation")
    for k in range(1, len(fence) - 1, 2):
        if fence[k + 1] == fence[k] + 1:
            F("L11", f"line {fence[k] + 1}: closing '$$' directly followed by opening '$$'; insert a blank line (renders as '$$$$')")
    # L7 / L8 / L9 per knowledge point
    kp_idx = [(i, h) for i, (lvl, h, _) in enumerate(heads) if lvl == 2 and re.match(r"KP\s*\d+", h)]
    for n, (i, h) in enumerate(kp_idx):
        start = heads[i][2]
        end = heads[kp_idx[n + 1][0]][2] if n + 1 < len(kp_idx) else len(text)
        block = text[start:end]
        if args.lang == "en":
            found = [zh_ for zh_, en_ in FIELDS if en_.lower() in block.lower()]
        else:
            found = [zh_ for zh_, en_ in FIELDS if f"**{zh_}**" in block or zh_ in block]
        miss = [zh_ for zh_, _ in FIELDS if zh_ not in found]
        if len(found) < 3: F("L7", f"{h[:20]!r}: only {len(found)}/6 SKILL 4.6 fields present (missing {miss})")
        elif miss: W("L7", f"{h[:20]!r}: missing fields {miss}")
        subs = [x for x in heads[i + 1:(kp_idx[n + 1][0] if n + 1 < len(kp_idx) else len(heads))] if x[0] == 3]
        if len(subs) > args.max_sections_per_kp:
            W("L8", f"{h[:20]!r}: {len(subs)} sections; a lecture explains in connected prose, it does not transcribe one slide per section")
    secs = re.split(r"^### .*$", text, flags=re.M)[1:]
    titles = [h for lvl, h, _ in heads if lvl == 3]
    tiny = [t for t, b in zip(titles, secs) if len(re.sub(r"[\s>*#|$-]", "", b)) < 40]
    if len(tiny) > 3: W("L9", f"{len(tiny)} sections are one-line fragments, e.g. {tiny[:3]}")


    # ---------------- readability checks L12-L22 ----------------
    import json, os
    en = args.lang == "en"
    body = re.sub(r"```.*?```", "", text, flags=re.S)
    body = re.sub(r"\$\$.*?\$\$", " ", body, flags=re.S)
    cap_re = re.compile(r"^(圖說|Figure|Caption)\s*`[^`]*`[:：].*$", re.M)
    captions = cap_re.findall(body)
    caps_full = [m.group(0) for m in cap_re.finditer(body)]
    prose = re.sub(r"^## (?:學習目標|Learning objectives).*?(?=^## )", "", body, flags=re.S | re.M)
    prose = cap_re.sub("", prose)
    prose = re.sub(r"!\[.*?\]\(.*?\)", "", prose)
    prose = re.sub(r"^\|.*$", "", prose, flags=re.M)
    prose = re.sub(r"^!!!.*$", "", prose, flags=re.M)
    prose = re.sub(r"^\[[^\]]*\]\([^)]*\).*$", "", prose, flags=re.M)
    prose = re.sub(r"^#+ .*$", "", prose, flags=re.M)
    plain = re.sub(r"\$[^$\n]*\$", "□", prose)
    plain = re.sub(r"[*`]", "", plain)
    paras = [p_.strip() for p_ in re.split(r"\n\s*\n", plain) if p_.strip() and not p_.strip().startswith(">")]
    split_s = (lambda t: [x for x in re.split(r"(?<=[.!?])\s+", t) if x.strip()]) if en else (lambda t: [x for x in re.split(r"[。；！？\n]", t) if x.strip()])
    sents = [x.strip() for p_ in paras for x in split_s(p_)]
    sents = [x for x in sents if not x.startswith(("- ", "1.", "2.", "3.")) or len(x) > 0]
    # L12
    if en: longs = [x for x in sents if len(x.split()) > 25]
    else: longs = [x for x in sents if len(x) > 40]
    if len(longs) >= 5: W("L12", f"{len(longs)} of {len(sents)} sentences are too long ({'over 25 words' if en else 'over 40 characters'}); split them. e.g. {longs[0][:60]!r}")
    # L13
    long_p = [p_ for p_ in paras if len(split_s(p_)) > 4]
    if long_p:
        ratio = len(long_p) / max(len(paras), 1)
        (F if ratio > 0.4 else W)("L13", f"{len(long_p)} of {len(paras)} paragraphs have more than 4 sentences (one idea per paragraph); e.g. {long_p[0][:50]!r}")
    # L14
    n_inline = len(re.findall(r"\$[^$\n]+\$", text))
    has_symtab = bool(re.search(r"符號表|符號說明|Symbol table|Notation|\|\s*(符號|Symbol)\s*\|", text))
    if n_inline >= 20 and not has_symtab: W("L14", f"{n_inline} inline formulas but no symbol table (symbol, meaning, dimension, where defined)")
    # L15 glossary
    gl = text.split("## 術語與來源")[-1] if "## 術語與來源" in text else (text.split("## Terms and source")[-1] if "## Terms and source" in text else "")
    gl_terms = re.findall(r"^\|\s*([^|\n]+?)\s*\|", gl, flags=re.M)
    gl_text = gl
    bold_terms = [t.strip() for t in re.findall(r"\*\*([^*\n]{2,30})\*\*\s*[:：]", body.split("## 術語與來源")[0].split("## Terms and source")[0])]
    stop = re.compile(r"^(白話說明|一句話重點|正式定義|例子|常見迷思|小練習|題|做法|結果|意思|設定|Plain|One-sentence|Formal|Example|Common|Practice|Question|Method|Results?|Meaning|Setup|Check|核對|怎麼記|To remember|實驗細節|Experiment details|答|解)")
    bold_terms = [t for t in dict.fromkeys(bold_terms) if not stop.match(t) and not re.match(r"^[□\s\d]+$", t)]
    miss_b = [t for t in bold_terms if t not in gl_text]
    acr = set(re.findall(r"\b[A-Z][A-Za-z]*[A-Z]{1,}[A-Za-z]*\b|\b[A-Z]{2,6}\b", re.sub(r"\$[^$]*\$", "", body.split("## 術語與來源")[0].split("## Terms and source")[0])))
    acr -= {"KP", "XOR", "AND", "OR", "NOT", "TP", "FP", "FN", "TN", "RBF", "SVM", "MLP", "II", "PDF", "CART", "LMS", "IRLS", "KKT", "RKHS", "SVR"} if False else {"KP", "AND", "OR", "NOT", "PDF", "XOR", "SVM", "AdaBoost", "RBF", "MLP", "PNN", "MSE", "EM"}
    miss_a = sorted(a for a in acr if a not in gl_text)
    if bold_terms and len(miss_b) / len(bold_terms) > 0.4: W("L15", f"glossary lists only {len(bold_terms)-len(miss_b)}/{len(bold_terms)} bold-defined terms; missing e.g. {miss_b[:4]}")
    if len(miss_a) >= 2: W("L15", f"acronyms not in the glossary: {miss_a[:8]}")
    # L16 practice vs solutions
    pm = re.search(r"(?:\*\*小練習\*\*|\*\*Practice\*\*)[^\n]*\n(.*?)(?=\n圖說|\nFigure|\n\*\*|\Z)", text, flags=re.S)
    if pm:
        qs = re.findall(r"^\s*(\d+)[.)]", pm.group(1), flags=re.M)
        sol = " ".join(re.findall(r"^\*\*([^*\n]*(?:題|[Qq]uestion|[Ss]olution)[^*\n]*)\*\*", text, flags=re.M))
        covered = set()
        for grp in re.findall(r"(?:題|[Qq]uestions?)\s*([\d、, and&]+)", sol): covered |= set(re.findall(r"\d+", grp))
        if not re.search(r"解題示範|solution|Solution|worked", text): F("L16", "practice given but no worked solution anywhere")
        elif qs and set(qs) - covered: F("L16", f"practice question(s) {sorted(set(qs)-covered)} have no worked solution (solutions cover {sorted(covered)})")
    # L17 deck overlap
    deck = args.deck
    if not deck:
        stem = os.path.basename(args.lecture)[:-3]
        stem = re.sub(r"-(zh|en)$", "", stem)
        for base in (os.path.join(os.path.dirname(os.path.abspath(args.lecture)), "..", "slides"), os.path.join(os.path.dirname(os.path.abspath(args.lecture)), "..", "..", "slides")):
            for cand in ([stem + "-en.jbd.json", stem + ".jbd.json"] if en else [stem + ".jbd.json"]):
                pth = os.path.join(base, cand)
                if os.path.exists(pth): deck = pth; break
            if deck: break
    if deck and os.path.exists(deck):
        try:
            d = json.load(open(deck, encoding="utf-8"))
            dt = re.sub(r"\s+", "", " ".join(str(n.get("text") or n.get("content") or "") for n in d["nodes"].values()))
            dt = re.sub(r"\$[^$]*\$", "", dt)
            if en:
                dw = " ".join(re.sub(r"\s+", " ", " ".join(str(n.get("text") or n.get("content") or "") for n in d["nodes"].values())).split())
                dtt = re.sub(r"\$[^$]*\$", " ", dw).lower(); dtt = " ".join(dtt.split()); cand = [" ".join(x.replace("□", " ").split()) for x in sents if len(x.split()) >= 8]
                hit = [x for x in cand if " ".join(x.lower().split()[:8]) in dtt]
            else:
                cand = [re.sub(r"[\s□]+", "", x) for x in sents if len(re.sub(r"[\s□]+", "", x)) >= 14]
                hit = [x for x in cand if x[:14] in dt]
            if cand:
                r_ = len(hit) / len(cand)
                if r_ > 0.5: F("L17", f"{len(hit)}/{len(cand)} ({r_:.0%}) of the lecture sentences start like a slide sentence: the lecture is a transcription of the deck")
                elif r_ > 0.2: W("L17", f"{len(hit)}/{len(cand)} ({r_:.0%}) of the lecture sentences start like a slide sentence; write the lecture first and extract the deck from it")
        except Exception as e_:
            W("L17", f"could not read the deck {deck}: {e_}")
    # L18 captions
    if caps_full:
        marked = [c for c in caps_full if re.search(r"〔功能|\[Function|先看|Look first|First look", c)]
        if len(marked) < len(caps_full): W("L18", f"{len(caps_full)-len(marked)} of {len(caps_full)} figure captions have no reading guide or function mark (先看… 〔功能：1、3〕)")
    # L19 experiment setup
    perf = [m for ln in body.splitlines() if re.search(r"準確率|召回|精確率|誤差|錯誤率|accuracy|recall|precision|error rate", ln, flags=re.I) for m in re.findall(r"\d+(?:\.\d+)?\s*%", ln)]
    if len(perf) >= 3 and not re.search(r"種子|seed|重複|repeat|runs|測試集|test set|新點|新資料|fresh", body, flags=re.I):
        W("L19", f"{len(perf)} performance numbers but no experiment setup words (seed, repeats, test set); see CLARITY.md C19")
    elif len(perf) >= 3 and not re.search(r"種子|seed", body, flags=re.I):
        W("L19", f"{len(perf)} performance numbers; no random seed is stated, so the numbers cannot be reproduced (CLARITY.md C19)")
    # L20 skipped steps
    skip = re.findall(r"顯然|易知|不難看出|容易看出|容易得到|由此可得|同理可得|obviously|clearly|it is easy to see|trivially", body, flags=re.I)
    if skip: W("L20", f"skipped-step wording {sorted(set(skip))}: write out the step")
    # L21 teaching text length
    n_zh = len(re.findall(r"[一-鿿]", plain)); n_w = len(re.findall(r"[A-Za-z]+", plain))
    if (en and n_w < 700) or (not en and n_zh < 1200): W("L21", f"teaching text is short ({n_w} words)" if en else f"teaching text is short ({n_zh} Chinese characters, math and captions excluded; at least 1200 expected for one knowledge point)")
    # L22 term used before its definition
    defs = [(m.group(1).strip(), m.start()) for m in re.finditer(r"\*\*([^*\n]{2,20})\*\*\s*[:：]", text)]
    exempt_end = 0
    mm = re.search(r"^### ", text, flags=re.M)
    if mm: exempt_end = mm.start()
    early = []
    for t, pos in defs:
        if stop.match(t) or pos <= exempt_end or re.search(r"[□$]", t): continue
        first = text.find(t, exempt_end)
        if 0 <= first < pos and not re.search(r"^### ", text[first:first + 1]):
            line_start = text.rfind("\n", 0, first) + 1
            if text[line_start:line_start + 3] == "###": continue
            early.append(t)
    if early: W("L22", f"term(s) used before their definition: {early[:5]}")

    if args.source:
        cj = re.compile(r"[㐀-鿿豈-﫿]")
        stream = lambda t: "".join(cj.findall(re.sub(r"\$[^$]*\$", "", t)))
        grams = set()
        for sp in args.source:
            st = stream(open(sp, encoding="utf-8").read())
            grams |= {st[i:i + 14] for i in range(len(st) - 13)}
        lt = stream(text)
        hits = sum(1 for i in range(len(lt) - 13) if lt[i:i + 14] in grams)
        if hits > 30: W("L10", f"{hits} positions start a 14-character run identical to the source: rewrite in your own words")
    for x in fails + warns: print(x)
    print(f"audit_lecture: {args.lecture}  headings {len(heads)}  knowledge points {len(kp_idx)}  FAIL {len(fails)}  WARN {len(warns)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
