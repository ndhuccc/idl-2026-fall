"""Write a unit's figure specs: make_specs.py UNIT_DIR KPNN N --lecture lecNN."""
import argparse
import json
import os
from pathlib import Path
import re
from course_context import lecture_prefix, repo_root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("unit_dir")
    parser.add_argument("kp")
    parser.add_argument("pages", type=int)
    parser.add_argument("--lecture", help="Defaults to course-unit.json; required for legacy units")
    parser.add_argument("--repo")
    args = parser.parse_args(argv)
    directory = Path(args.unit_dir).expanduser().resolve()
    kp = args.kp.lower()
    if not re.fullmatch(r"kp\d+", kp) or args.pages < 1:
        parser.error("Use KPNN and a positive page count")
    config_file = directory / "course-unit.json"
    config = json.loads(config_file.read_text(encoding="utf-8")) if config_file.is_file() else {}
    value = args.lecture or config.get("lecture") or os.environ.get("LEC")
    if not value:
        parser.error("--lecture is required when course-unit.json is absent")
    lecture = lecture_prefix(value)
    repo = repo_root(args.repo or os.environ.get("COURSE_REPO") or config.get("repo_root"))
    # Keep the selected repository in editable unit metadata, not generated source.
    config.update(repo_root=str(repo), lecture=lecture)
    config_file.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    source = f'''import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
from course_context import load_unit
CONTEXT = load_unit(__file__)
sys.path.insert(0, str(HERE))
import {kp}_pages as KP
from htmlkit import native
JD = str(HERE / "out")
def _f(n):
    return lambda: native(f"{kp}_p{{n}}", f"{{CONTEXT['lecture']}}-{kp}-p{{n}}", KP.W, KP.H, JD, "figure")
{", ".join(f"p{i}" for i in range(1, args.pages + 1))}, = (_f(i) for i in range(1, {args.pages + 1}))
'''
    import shutil
    shutil.copy(Path(__file__).with_name("course_context.py"), directory)
    (directory / f"{kp}_specs.py").write_text(source, encoding="utf-8")
    print("wrote", f"{kp}_specs.py")


if __name__ == "__main__":
    main()
