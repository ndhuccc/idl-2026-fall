#!/usr/bin/env python3
"""insert_page.py -- put a page built with jbd_page.Page into a JianBauDau deck (new or existing).

    python3 insert_page.py deck.jbd.json --nodes nodes.json --after 3 --label "CNN vs flatten" --notes "speaker notes ..."
    python3 insert_page.py deck.jbd.json --nodes nodes.json --replace 2 --label ...
    python3 insert_page.py deck.jbd.json --nodes nodes.json --after 0          # first page of an empty deck

Pages are numbered 1.. in the child scene; page n sits at x = (n-1)*1440 and has camera step n (step 0 is the root step).
--after N : new page becomes N+1; every later page, its nodes and its step move one slot right (1440 px).
--replace N: the nodes of page N are removed and replaced; its step keeps label/notes unless you pass new ones.
Writes in place (use --out to write elsewhere). Prints the new page number. Never touches other pages' content."""
import argparse, copy, json, re

PITCH = 1440


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("deck"); ap.add_argument("--nodes", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--after", type=int); g.add_argument("--replace", type=int)
    ap.add_argument("--label", default=None); ap.add_argument("--notes", default=None); ap.add_argument("--scene", default=None, help="child scene id (default: the only non-root scene)")
    ap.add_argument("--prefix", default=None, help="node id prefix (default: automatically chosen unused prefix)"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = json.load(open(a.deck, encoding="utf-8")); N, S, st = d["nodes"], d["scenes"], d["steps"]
    sid = a.scene or [k for k in S if k != d["rootSceneId"]][0]; sc = S[sid]
    new = json.load(open(a.nodes, encoding="utf-8"))["nodes"]
    if a.replace is not None:
        n = a.replace; x0 = (n - 1) * PITCH
        gone = [k for k in sc["nodeIds"] if x0 - 1 <= N[k]["transform"]["x"] < x0 + PITCH - 100]
        for k in gone: N.pop(k, None)
        sc["nodeIds"] = [k for k in sc["nodeIds"] if k not in gone]
        step = st[n]
        if a.label: step["label"] = a.label
        if a.notes is not None: step["speakerNotes"] = a.notes
        at = len(sc["nodeIds"])
    else:
        n = a.after + 1; x0 = a.after * PITCH
        for k, v in N.items():
            if v["sceneId"] == sid and v["transform"]["x"] >= x0 - 1: v["transform"]["x"] += PITCH
        for s in st[a.after + 1:]: s["target"]["point"]["x"] += PITCH
        ref = st[a.after] if a.after >= 1 else (st[1] if len(st) > 1 else None)
        ns = copy.deepcopy(ref) if ref else {"durationMs": 900, "holdMs": 500, "easing": "smooth",
                                             "target": {"kind": "scene-point", "sceneId": sid, "point": {"x": 0, "y": 360.0}, "zoom": 1.0, "referenceViewport": {"width": 1280, "height": 720}}}
        ns = {k: v for k, v in ns.items() if k in ("id", "order", "label", "durationMs", "holdMs", "easing", "target", "speakerNotes")}   # never copy narration audio etc. from the reference step
        ns["id"] = f"{sid}-step-new{len(st)}"; ns["label"] = a.label or f"Page {n}"; ns["target"]["point"]["x"] = x0 + 640
        ns["target"]["point"]["y"] = 360.0
        if a.notes is not None: ns["speakerNotes"] = a.notes
        else: ns.pop("speakerNotes", None)
        st.insert(a.after + 1, ns)
        for i, s in enumerate(st): s["order"] = i
        at = len(sc["nodeIds"])
    pre = a.prefix or f"{sid}-pnew{n}"
    if a.prefix is None:
        base, suffix = pre, 1
        while any(f"{pre}-{i + 1}" in N for i in range(len(new))):
            pre = f"{base}-v{suffix}"
            suffix += 1
    for i, nd in enumerate(new):
        nd = copy.deepcopy(nd); nd["id"] = f"{pre}-{i + 1}"; nd["sceneId"] = sid; nd["transform"]["x"] = round(nd["transform"]["x"] + x0, 2)
        if nd["id"] in N: raise SystemExit(f"node id clash: {nd['id']} (use --prefix)")
        N[nd["id"]] = nd; sc["nodeIds"].append(nd["id"])
    json.dump(d, open(a.out or a.deck, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"page {n} written ({len(new)} nodes) -> {a.out or a.deck}")


if __name__ == "__main__":
    main()
