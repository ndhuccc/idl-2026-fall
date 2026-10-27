"""Resolve the current course and unit without machine-specific paths."""
import json
import os
import re
import sys
from pathlib import Path


def repo_root(value=None):
    selected = value or os.environ.get("COURSE_REPO")
    if selected:
        root = Path(selected).expanduser().resolve()
        if not (root / ".agents" / "skills").is_dir():
            raise ValueError(f"Not a course repository: {root}")
        return root
    for parent in Path(__file__).resolve().parents:
        if (parent / ".agents" / "skills").is_dir():
            return parent
    raise ValueError("Set COURSE_REPO or pass --repo to select the course repository")


def lecture_prefix(value):
    match = re.match(r"^(lec\d+)(?:-|$)", value)
    if not match:
        raise ValueError("Lecture must start with lecNN (for example lec07)")
    return match.group(1)


def figure_prefix(stem):
    match = re.match(r"^(lec\d+-kp\d+)(?:-|$)", stem)
    if not match:
        raise ValueError("Lecture filename must start with lecNN-kpNN; or supply --prefix")
    return match.group(1)


def course_names(root):
    """Read configured names; never substitute another course's name."""
    settings = root / "semester.toml"
    zh, en = "", ""
    if settings.is_file():
        import tomllib
        data = tomllib.loads(settings.read_text(encoding="utf-8"))
        zh = data.get("course_name", "")
        en = data.get("course_name_en", data.get("extra", {}).get("course_name_en", ""))
    state = root / "COURSE_STATE.md"
    if not zh and state.is_file():
        match = re.search(r"^\|\s*課程\s*\|\s*([^|]+)\|", state.read_text(encoding="utf-8"), re.M)
        if match:
            zh = re.split(r"[（(]", match.group(1).strip(), maxsplit=1)[0].strip()
    return zh, en


def load_unit(filename):
    """A generated deck can run in its unit folder or a W_zh/W_en subfolder."""
    for parent in Path(filename).resolve().parents:
        config = parent / "course-unit.json"
        if config.is_file():
            data = json.loads(config.read_text(encoding="utf-8"))
            root = repo_root(os.environ.get("COURSE_REPO") or data["repo_root"])
            skills = root / ".agents" / "skills"
            os.environ["COURSE_REPO"] = str(root)
            os.environ["COURSE_SKILLS"] = str(skills)
            os.environ["SKILLS_DIR"] = str(skills)
            for part in ("view-from-text/tools/deckkit", "view-from-text-v2/tools",
                         "view-from-text-v2/examples", "view-from-text-v3/tools"):
                sys.path.insert(0, str(skills / part))
            sys.path.insert(0, str(parent))
            data["repo_root"] = str(root)
            return data
    raise ValueError(f"No course-unit.json found above {filename}")
