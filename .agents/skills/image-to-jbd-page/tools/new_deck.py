#!/usr/bin/env python3
"""new_deck.py -- write an empty JianBauDau v3 deck (root canvas + one frame + one scene, no pages yet).

    python3 new_deck.py out.jbd.json --title "My deck"

Pages are then added with insert_page.py. The layout (root frame -> child scene, pages 1440 px apart, one camera step per page)
is the same one the course decks use, so insert_page.py works on both new and existing decks."""
import argparse, json


def make(title="Untitled", name="Page", uid="u01"):
    sid, fid = f"scene-{uid}", f"frame-{uid}"
    return {"schemaVersion": 3, "rootSceneId": "scene-root",
            "scenes": {"scene-root": {"id": "scene-root", "name": "Root", "parentFrameId": None, "nodeIds": [fid], "camera": {"x": 200, "y": 100, "zoom": 1.2, "rotation": 0}},
                       sid: {"id": sid, "name": name, "parentFrameId": fid, "nodeIds": [], "camera": {"x": 640.0, "y": 360.0, "zoom": 1.0, "rotation": 0}}},
            "nodes": {fid: {"id": fid, "type": "frame", "name": fid, "sceneId": "scene-root", "transform": {"x": 0, "y": 0, "scaleX": 1, "scaleY": 1, "rotation": 0},
                            "width": 400, "height": 200, "zIndex": 1, "locked": False,
                            "style": {"fill": "transparent", "stroke": "transparent", "textColor": "#f2ede4", "fontSize": 22, "radius": 0}, "text": name,
                            "frame": {"childSceneId": sid, "previewMode": "live", "padding": 18, "clip": True}}},
            "steps": [{"id": f"root-step-{uid}", "order": 0, "label": name, "durationMs": 900, "holdMs": 500, "easing": "smooth",
                       "target": {"kind": "scene-point", "sceneId": "scene-root", "point": {"x": 200, "y": 100}, "zoom": 8 / 3, "referenceViewport": {"width": 1280, "height": 720}}}],
            "metadata": {"title": title, "createdAt": "2026-01-01T00:00:00.000Z", "updatedAt": "2026-01-01T00:00:00.000Z", "aspectRatio": "16:9"}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--title", default="Untitled"); a = ap.parse_args()
    json.dump(make(a.title), open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2); print("wrote", a.out)
