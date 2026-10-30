#!/usr/bin/env python3
"""Quality gate for a generated JianBauDau deck (.jbd.json made from page-to-teaching-diagram pages).

Checks (see SKILL.md, quality gates 6 and 8, and "課堂簡報製作流程"):
  1. the opening step shows a KP short title (not the default "Page");
  2. layer check: no text node whose centre lies inside a higher-zIndex, opaque filled shape of the same page;
  3. figure-first: per page, count figure elements (star, polygon, polyline, triangle, diamond, ellipse wider than 14 px);
     a page with fewer than --min-figures is reported (cards-only pages must be reworked).
  4. lively elements: per page, count how many kinds of lively effects appear (gradient, shadow/glow, star, polyline/polygon,
     dashed line, rotated sticker/stamp, mascot-sized ellipse group); pages with fewer than --min-lively kinds are reported;
     the deck must contain at least one rotated sticker/stamp.
  5. reveal (progressive appearance): every initiallyHidden node is revealed by some step; every revealed node exists, is
     initiallyHidden and belongs to the step's scene; no node is revealed twice; per page at least --min-beats reveal steps;
     every reveal step has speaker notes.
Usage:  python3 check_deck_quality.py deck.jbd.json [--min-figures 4] [--page-width 1280]
  6. feature coverage (see references/feature-catalog.md): the deck should use native connectors, in-page camera moves, inline
     text formatting, reveal exit, multi-stop gradients, fill opacity, and other JianBauDau features; reports which of the
     catalog's feature groups are used / missing and warns below --min-features groups.
  7. native connectors: connectors are counted as figure elements; the deck should use at least --min-connectors of them, and
     hand-drawn arrows (a polyline with a small triangle arrowhead at its end) are listed so flow pages can be converted.
  8. bubbles and text boxes: every white speech/thought bubble must hold a non-empty text node above it and that text must be
     revealed in the same step as the bubble; text nodes with empty text and nodes sticking out of the 720 px page are reported.
Exit code 1 if check 1, 2, 5 (reveal structure) or 8 (empty or out-of-sync bubble) fails; checks 3, 4, 7 and the rest of 8 only warn (exit 0) unless --strict.
"""
import argparse, collections, json, re, sys

def alpha(c):
    if not c or c == "transparent": return 0.0
    m = re.match(r"rgba\(([^)]+)\)", c.strip())
    if m:
        p = [x.strip() for x in m.group(1).split(",")]
        return float(p[3]) if len(p) > 3 else 1.0
    return 1.0

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("deck"); ap.add_argument("--min-figures", type=int, default=4)
    ap.add_argument("--page-width", type=int, default=1280); ap.add_argument("--min-lively", type=int, default=3); ap.add_argument("--min-beats", type=int, default=2); ap.add_argument("--min-features", type=int, default=9); ap.add_argument("--min-connectors", type=int, default=4); ap.add_argument("--strict", action="store_true")
    a = ap.parse_args(); d = json.load(open(a.deck, encoding="utf-8")); bad = 0
    steps = sorted(d["steps"], key=lambda s: s["order"])
    first = (steps[0].get("label") or "").strip() if steps else ""
    if not first or first.lower().startswith("page"): print("FAIL  opening step label is empty or 'Page':", repr(first)); bad += 1
    else: print("ok    opening step label:", first)
    # pages = columns of nodes around each scene-point step target (pages are laid side by side in one scene)
    centers = collections.defaultdict(set)
    for s_ in steps:
        t_ = s_["target"]
        if t_.get("point") and t_["sceneId"] != d["rootSceneId"]: centers[t_["sceneId"]].add(round(t_["point"]["x"]))
    cl = {sc: sorted(v) for sc, v in centers.items()}
    def page_of(sc, x):
        c = cl.get(sc)
        return min(range(len(c)), key=lambda i: abs(c[i] - x)) if c else int(x // a.page_width)
    pages = collections.defaultdict(list)
    for n in d["nodes"].values():
        if n["sceneId"] == d["rootSceneId"] or n["type"] == "connector": continue
        pages[(n["sceneId"], page_of(n["sceneId"], n["transform"]["x"] + n["width"] / 2))].append(n)
    conn_per_page = collections.Counter()
    for c_ in (n for n in d["nodes"].values() if n["type"] == "connector"):
        src = d["nodes"].get((c_.get("connector") or {}).get("sourceNodeId"))
        if src and src["sceneId"] != d["rootSceneId"]: conn_per_page[(src["sceneId"], page_of(src["sceneId"], src["transform"]["x"] + src["width"] / 2))] += 1
    z_bad = []
    for key, ns in pages.items():
        fills = [n for n in ns if n["type"] == "shape" and alpha(n["style"].get("fill")) >= 0.9 and n.get("shapeKind") != "polyline"]
        for t in ns:
            if not (t.get("text") or "").strip(): continue
            cx = t["transform"]["x"] + t["width"] / 2; cy = t["transform"]["y"] + t["height"] / 2
            for f in fills:
                if f is t or f["zIndex"] <= t["zIndex"]: continue
                fx, fy = f["transform"]["x"], f["transform"]["y"]
                if fx <= cx <= fx + f["width"] and fy <= cy <= fy + f["height"]: z_bad.append((key, t["text"][:30], t["zIndex"], f["zIndex"])); break
    if z_bad:
        bad += 1; print("FAIL  text under a higher opaque shape: %d" % len(z_bad)); [print("      ", z) for z in z_bad[:10]]
    else: print("ok    no text hidden under higher shapes")
    low = []
    for key in sorted(pages):
        ns = pages[key]
        if len(ns) < 8: continue          # stray decoration outside the page columns
        fig = sum(1 for n in ns if n["type"] == "shape" and (n.get("shapeKind") in ("star", "polygon", "polyline", "triangle", "diamond") or (n.get("shapeKind") == "ellipse" and n["width"] > 14)))
        fig += conn_per_page.get(key, 0)
        if fig < a.min_figures: low.append((key[1] + 1, fig))
    print(("WARN  pages with fewer than %d figure elements (page, count): %s" % (a.min_figures, low)) if low else "ok    every page has enough figure elements")
    if low and a.strict: bad += 1
    # ---- 4. lively elements
    def kinds(ns):
        k = set()
        for n in ns:
            st = n.get("style", {}); sk = n.get("shapeKind")
            if st.get("gradientTo"): k.add("gradient")
            if st.get("shadowColor"): k.add("shadow")
            if sk == "star": k.add("star")
            if sk in ("polyline", "polygon"): k.add("curve")
            if st.get("borderStyle") in ("dashed", "dotted"): k.add("dashed")
            if n["transform"].get("rotation"): k.add("rotated")
            if sk == "ellipse" and n["width"] > 14: k.add("ellipse")
        return k
    dull = [(key[1] + 1, len(kinds(ns))) for key, ns in sorted(pages.items()) if len(ns) >= 8 and len(kinds(ns)) < a.min_lively]
    print(("WARN  pages with fewer than %d kinds of lively effects (page, kinds): %s" % (a.min_lively, dull)) if dull else "ok    every page has varied lively effects")
    if not any(n["transform"].get("rotation") for n in d["nodes"].values() if n["sceneId"] != d["rootSceneId"]): print("WARN  no rotated sticker/stamp anywhere in the deck"); dull.append("no-sticker")
    if dull and a.strict: bad += 1
    # ---- 5. reveal
    hidden = {i for i, n in d["nodes"].items() if n.get("initiallyHidden")}; seen = collections.Counter(); rv_bad = []; beats = collections.Counter(); no_notes = 0
    for s_ in steps:
        r = s_.get("reveal")
        if not r: continue
        ids = [e["nodeId"] for e in r.get("enter", [])]
        for i in ids:
            seen[i] += 1
            if i not in d["nodes"]: rv_bad.append(("missing node", i))
            else:
                if d["nodes"][i]["sceneId"] != s_["target"]["sceneId"]: rv_bad.append(("other scene", i))
                if i not in hidden: rv_bad.append(("not initiallyHidden", i))
        for i in r.get("exitNodeIds", []):
            if i not in d["nodes"] or d["nodes"][i]["sceneId"] != s_["target"]["sceneId"]: rv_bad.append(("bad exit node", i))
        if ids:
            n0 = d["nodes"].get(ids[0]); 
            if n0: beats[(n0["sceneId"], page_of(n0["sceneId"], n0["transform"]["x"] + n0["width"] / 2))] += 1
            if not (s_.get("speakerNotes") or "").strip(): no_notes += 1
    never = sorted(hidden - set(seen)); twice = [i for i, c in seen.items() if c > 1]
    if never: rv_bad.append(("hidden but never revealed", len(never)))
    if twice: rv_bad.append(("revealed more than once", len(twice)))
    if no_notes: rv_bad.append(("reveal steps without speaker notes", no_notes))
    if rv_bad: bad += 1; print("FAIL  reveal problems:", rv_bad[:8])
    else: print("ok    reveal structure is consistent (%d hidden nodes, all revealed once)" % len(hidden))
    few = [(key[1] + 1, beats.get(key, 0)) for key in sorted(pages) if len(pages[key]) >= 8 and beats.get(key, 0) < a.min_beats]
    print(("WARN  pages with fewer than %d reveal beats (page, beats): %s" % (a.min_beats, few)) if few else "ok    every page has enough reveal beats")
    if few and a.strict: bad += 1
    # ---- 6. feature coverage
    N = [n for n in d["nodes"].values() if n["sceneId"] != d["rootSceneId"]]
    cam = 0
    for p_, c_ in zip(steps, steps[1:]):
        t1, t2 = p_["target"], c_["target"]
        if t1["sceneId"] == t2["sceneId"] and t1.get("point") and t2.get("point") and abs(t2["point"]["x"] - t1["point"]["x"]) < 700 and (t1["point"] != t2["point"] or t1.get("zoom", 1) != t2.get("zoom", 1)): cam += 1
    cam += sum(1 for s_ in steps if s_["target"]["kind"] == "focus-node")
    feats = {
        "basic shapes (rect/ellipse/triangle/diamond/pentagon/star/arrow)": len({n.get("shapeKind") for n in N if n["type"] == "shape"} & {"triangle", "diamond", "pentagon", "star", "arrow"}) >= 3,
        "polygon / polyline (smooth curves)": any(n.get("shapeKind") in ("polygon", "polyline") and any(p.get("smooth") for p in n.get("polygonPoints", [])) for n in N),
        "native connectors": any(n["type"] == "connector" for n in d["nodes"].values()),
        "fill opacity (fillOpacity or rgba fill)": any(n.get("style", {}).get("fillOpacity") not in (None, 1) or 0 < alpha(n.get("style", {}).get("fill")) < 0.95 for n in N),
        "gradient (linear/radial)": any(n.get("style", {}).get("gradientTo") for n in N),
        "multi-stop gradient": any(n.get("style", {}).get("gradientStops") for n in N),
        "shadow / glow": any(n.get("style", {}).get("shadowColor") for n in N),
        "line styles (dashed/dotted/double)": any(n.get("style", {}).get("borderStyle") in ("dashed", "dotted", "double") for n in N),
        "rotation / sticker": any(n["transform"].get("rotation") for n in N),
        "inline text formatting ({color=}/{size=})": any(re.search(r"\{(color|size|font)=", n.get("text") or "") for n in N),
        "reveal exit": any(s_.get("reveal", {}).get("exitNodeIds") for s_ in steps),
        "in-page camera moves / focus": cam > 0,
    }
    used = [k for k, v in feats.items() if v]; miss = [k for k, v in feats.items() if not v]
    print("info  feature groups used %d/%d" % (len(used), len(feats)))
    print(("WARN  missing: " + "; ".join(miss)) if miss else "ok    all feature groups used")
    if len(used) < a.min_features:
        print("WARN  fewer than %d feature groups used (see references/feature-catalog.md)" % a.min_features)
        if a.strict: bad += 1
    # ---- 7. native connectors
    nconn = sum(1 for n in N if n["type"] == "connector")
    tri = [n for n in N if n["type"] == "shape" and n.get("shapeKind") in ("triangle", "arrow", "polygon") and n["width"] <= 30 and n["height"] <= 30]
    hand = 0
    for p_ in (n for n in N if n["type"] == "shape" and n.get("shapeKind") == "polyline"):
        x0, y0 = p_["transform"]["x"], p_["transform"]["y"]
        if any(t_["sceneId"] == p_["sceneId"] and x0 - 20 <= t_["transform"]["x"] + t_["width"] / 2 <= x0 + p_["width"] + 20 and y0 - 20 <= t_["transform"]["y"] + t_["height"] / 2 <= y0 + p_["height"] + 20 for t_ in tri): hand += 1
    print("info  native connectors: %d; hand-drawn arrows (polyline + small triangle head): %d" % (nconn, hand))
    if nconn < a.min_connectors:
        print("WARN  fewer than %d native connectors: draw flows, causes and correspondences with connector nodes (feature-catalog.md)" % a.min_connectors)
        if a.strict: bad += 1
    if hand: print("WARN  %d hand-drawn arrows: convert flow-like ones to native connectors" % hand)
    # ---- 8. bubbles, empty texts, page bounds
    when = {}
    for s_ in steps:
        for e in (s_.get("reveal") or {}).get("enter", []): when[e["nodeId"]] = s_["order"]
    bub = []; small = []
    def est_h(m):    # conservative text height: CJK 1.15 em, Latin 0.65 em per character, line height 1.3
        fs_ = m["style"].get("fontSize", 16); L = 0
        for ln in (m.get("text") or "").split("\n"):
            L += max(1, -(-int(sum(fs_ * 1.15 if ord(c) > 255 else fs_ * 0.65 for c in ln.replace("$", ""))) // max(1, int(m["width"] - 8))))
        return L * fs_ * 1.3
    for n in N:
        st = n.get("style", {}); t = n["transform"]
        if n["type"] == "shape" and st.get("radius", 0) >= 16 and str(st.get("fill", "")).lower() == "#ffffff" and st.get("shadowColor") and 60 < n["width"] < 700 and 30 < n["height"] < 200:
            ins = [m for m in N if m["type"] == "text" and m["sceneId"] == n["sceneId"] and m["zIndex"] > n["zIndex"] and (m.get("text") or "").strip() and t["x"] - 12 <= m["transform"]["x"] and m["transform"]["x"] + m["width"] <= t["x"] + n["width"] + 12 and t["y"] - 12 <= m["transform"]["y"] and m["transform"]["y"] + m["height"] <= t["y"] + n["height"] + 12]
            if not ins: bub.append(("empty bubble", n["id"], round(t["x"]), round(t["y"])))
            elif len(ins) == 1 and "$" not in ins[0]["text"] and est_h(ins[0]) + 20 > n["height"]: small.append((n["id"], round(n["height"]), round(max(est_h(m) for m in ins) + 26)))
            elif any(when.get(m["id"]) != when.get(n["id"]) for m in ins): bub.append(("text revealed in a different step than its bubble", n["id"]))
    if bub: bad += 1; print("FAIL  bubble problems:", bub[:8])
    else: print("ok    every white bubble holds text revealed with it")
    print(("WARN  %d bubbles too small for their text (id, height, needed): %s" % (len(small), small[:6])) if small else "ok    every bubble is large enough for its text")
    if small and a.strict: bad += 1
    # thought clouds: three white ellipses (zIndex 7, bordered) + text (zIndex 12) 8 ids later; the text must fit the cloud
    cloud_small = []
    for m in N:
        if m["type"] != "text" or m["zIndex"] != 12: continue
        mt = re.match(r"(.*-)(\d+)$", m["id"])
        if not mt: continue
        outs = [d["nodes"].get(mt.group(1) + str(int(mt.group(2)) - 8 + i)) for i in range(3)]
        if all(o and o.get("shapeKind") == "ellipse" and o["zIndex"] == 7 and o["style"].get("borderWidth", 0) > 0 for o in outs):
            top = min(o["transform"]["y"] for o in outs); hh = max(o["transform"]["y"] + o["height"] for o in outs) - top
            if est_h(m) + 14 > hh * 0.6 + 8: cloud_small.append((m["id"], round(hh), round(est_h(m) / 0.6 + 14)))
    print(("WARN  %d thought clouds too small for their text (id, height, needed): %s" % (len(cloud_small), cloud_small[:6])) if cloud_small else "ok    every thought cloud is large enough for its text")
    if cloud_small and a.strict: bad += 1
    empty = [n["id"] for n in N if n["type"] == "text" and not (n.get("text") or "").strip()]
    print(("WARN  %d text nodes with empty text (a missing caption?): %s" % (len(empty), empty[:6])) if empty else "ok    no empty text nodes")
    out = [n["id"] for n in N if n["type"] != "connector" and n["zIndex"] > 0 and (n["transform"]["y"] < -4 or n["transform"]["y"] + n["height"] > 724) and n["height"] < 700]
    print(("WARN  %d nodes stick out of the 720 px page: %s" % (len(out), out[:6])) if out else "ok    nothing sticks out of the page")
    sys.exit(1 if bad else 0)
main()
