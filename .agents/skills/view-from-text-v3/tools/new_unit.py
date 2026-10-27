"""Scaffold a unit for the selected course: new_unit.py DIR KPNN ZH EN --lecture lecNN."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
from course_context import course_names, lecture_prefix, repo_root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("kp")
    parser.add_argument("zh")
    parser.add_argument("en")
    parser.add_argument("--lecture", default=os.environ.get("LEC"), help="Lecture prefix, for example lec07; LEC is a legacy alias")
    parser.add_argument("--repo", help="Course repository; defaults to this tool's repository")
    parser.add_argument("--course-zh")
    parser.add_argument("--course-en")
    args = parser.parse_args(argv)
    root = repo_root(args.repo)
    if not args.lecture:
        parser.error("--lecture is required (or set LEC explicitly)")
    lecture = lecture_prefix(args.lecture)
    kp = args.kp.lower()
    if not re.fullmatch(r"kp\d+", kp):
        parser.error("KP must have the form KPNN")
    directory = Path(args.directory).expanduser().resolve()
    if directory.exists() and any(directory.iterdir()):
        parser.error("Use an empty unit directory to avoid replacing existing work")
    zh, en = course_names(root)
    config = dict(repo_root=str(root), lecture=lecture,
                  course_zh=zh if args.course_zh is None else args.course_zh,
                  course_en=en if args.course_en is None else args.course_en)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "course-unit.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tools = root / ".agents/skills"
    shutil.copy(Path(__file__).with_name("course_context.py"), directory)
    shutil.copy(tools / "view-from-text-v2/tools/deckcommon.py", directory)
    for lang in ("zh", "en"):
        work = directory / f"W_{lang}"
        work.mkdir()
        (work / "figs.json").write_text("{}\n", encoding="utf-8")
    for name in ("terms.json", "terms_en.json"):
        (directory / name).write_text("{}\n", encoding="utf-8")
    deck = f'''import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from course_context import load_unit
CONTEXT = load_unit(__file__)
os.environ["DECK_PFX"] = CONTEXT["lecture"] + {kp!r}
from deckv2 import run
from {kp}_text import PAGES, UNIT_ZH, UNIT_EN, TITLE_ZH, TITLE_EN, NOTE_ZH, NOTE_EN
if __name__ == "__main__":
    run(PAGES, UNIT_ZH, UNIT_EN, TITLE_ZH, TITLE_EN, NOTE_ZH, NOTE_EN, str(HERE / "draft.jbd.json"))
'''
    text = f'''from course_context import load_unit
CONTEXT = load_unit(__file__)
UNIT_ZH, UNIT_EN = {args.zh!r}, {args.en!r}
TITLE_ZH = (CONTEXT["course_zh"] + " " if CONTEXT["course_zh"] else "") + {kp.upper()!r} + "：" + UNIT_ZH
TITLE_EN = (CONTEXT["course_en"] + " " if CONTEXT["course_en"] else "") + {kp.upper()!r} + ": " + UNIT_EN
NOTE_ZH = "教材程度：大學三年級。"
NOTE_EN = "Level: undergraduate (year 3)."
# Add approved page dictionaries here; one page has lab/title/bridge/notes pairs
# and layout/spec, as described in SKILL.md. Empty scaffolds are not finished decks.
PAGES = []
'''
    (directory / f"{kp}_deck.py").write_text(deck, encoding="utf-8")
    (directory / f"{kp}_text.py").write_text(text, encoding="utf-8")
    print("created", directory, "for", root.name, lecture)


if __name__ == "__main__":
    main()
