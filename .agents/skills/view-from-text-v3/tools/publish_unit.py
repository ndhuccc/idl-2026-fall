"""Publish one approved unit in the selected course.
--dry-run only prints a plan. Other repositories and optional logs are written
only when explicitly supplied with --sync-to and --log-file.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime
import glob
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from zoneinfo import ZoneInfo
from course_context import figure_prefix, lecture_prefix, repo_root


def run(repo, *args, check=True):
    result = subprocess.run(list(map(str, args)), cwd=repo, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    if check and result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {args}")
    return result


@contextmanager
def course_lock(repo):
    identity = hashlib.sha256(str(repo).encode("utf-8")).hexdigest()[:20]
    lock = Path(tempfile.gettempdir()) / f"jbd-publish-{identity}.lock"
    for attempt in range(120):
        try:
            lock.mkdir()
            break
        except FileExistsError:
            if attempt == 119:
                raise RuntimeError(f"Publish lock is busy: {lock}")
            time.sleep(1)
    try:
        yield
    finally:
        lock.rmdir()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("unit-dir", "kp", "pages", "lecture"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--map", default="map.json")
    parser.add_argument("--caps", default="caps.json")
    parser.add_argument("--lecture-figures", choices=("auto", "preserve", "regenerate"), default="auto")
    for flag in ("remove", "en-before", "changelog", "state", "log"):
        parser.add_argument("--" + flag, default="")
    parser.add_argument("--repo", help="Defaults to this tool's course repository")
    parser.add_argument("--log-file", help="Optional log path; relative paths are within the selected course")
    parser.add_argument("--sync-to", action="append", default=[], help="Explicit additional repository for example files")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    repo = repo_root(args.repo)
    unit = Path(args.unit_dir).expanduser().resolve()
    stem, kp = args.lecture, args.kp.lower()
    if not re.fullmatch(r"lec\d+-kp\d+(?:-[A-Za-z0-9_-]+)?", stem):
        parser.error("--lecture must be a filename stem such as lec07-kp03-topic")
    prefix = figure_prefix(stem)
    if prefix.split("-")[1] != kp:
        parser.error("--kp and --lecture refer to different units")
    if not args.pages.isdigit() or int(args.pages) < 1:
        parser.error("--pages must be a positive integer")
    zh_md, en_md = repo / f"lectures/{stem}.md", repo / f"lectures/en/{stem}.md"
    layout = "split" if en_md.is_file() else "bilingual"
    figures = args.lecture_figures
    if figures == "auto":
        figures = "regenerate" if layout == "split" else "preserve"
    if layout == "bilingual" and figures == "regenerate":
        parser.error("A bilingual lecture keeps approved figures; use --lecture-figures preserve")
    others = [repo_root(path) for path in args.sync_to]
    if repo in others or len(set(others)) != len(others):
        parser.error("--sync-to must name distinct additional repositories")
    log = Path(args.log_file).expanduser() if args.log_file else None
    if log is not None:
        log = (log if log.is_absolute() else repo / log).resolve()
    removed = [name for name in args.remove.split(",") if name]
    if any(Path(name).name != name or not name.startswith(prefix + "-") or not name.endswith(".png") for name in removed):
        parser.error("--remove must contain PNG basenames belonging to the selected unit")
    if figures == "preserve" and removed:
        parser.error("--remove cannot be used while preserving lecture figures")
    plan = dict(repo=str(repo), lecture=stem, prefix=prefix, lecture_layout=layout, lecture_figures=figures,
                slides=[f"slides/{stem}.jbd.json", f"slides/{stem}-en.jbd.json"],
                log_file=str(log) if log else None, sync_to=list(map(str, others)),
                remove=removed)
    # Return before creating locks, temporary files, running Git, or writing content.
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    tools = repo / ".agents/skills/view-from-text-v3/tools"
    sources = [unit / f"W_{lang}/draft.jbd.json" for lang in ("zh", "en")]
    sources += [unit / f"{kp}_pages.py", unit / f"{kp}_text.py", unit / f"{kp}_nums.py"]
    if figures == "regenerate":
        sources += [unit / args.map, unit / args.caps]
    for path in sources:
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    if not zh_md.is_file():
        parser.error("The approved lecture must exist in the selected course")
    for target in [repo] + others:
        if run(target, "git", "status", "--porcelain").stdout.strip():
            parser.error(f"Working tree is not clean: {target}")
    now = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M")
    with course_lock(repo), tempfile.TemporaryDirectory(prefix="jbd-publish-") as temporary:
        run(repo, "git", "pull", "--rebase", "origin", "main")
        for lang, suffix in (("zh", ""), ("en", "-en")):
            shutil.copy(unit / f"W_{lang}/draft.jbd.json", repo / f"slides/{stem}{suffix}.jbd.json")
        if figures == "regenerate":
            run(repo, sys.executable, tools / "make_specs.py", unit, kp, args.pages,
                "--repo", repo, "--lecture", lecture_prefix(stem))
            run(repo, sys.executable, "-W", "ignore", tools / "gen_figs.py",
                unit / f"{kp}_specs.py", unit / args.map, repo / "assets/figures")
            for name in removed:
                fid = name[:-4]
                for path in (zh_md, en_md):
                    text = path.read_text(encoding="utf-8")
                    pattern = r"(圖說|Figure) \x60%s\x60.*?%s(-en)?\.png\)\n\n?" % (re.escape(fid), re.escape(fid.split(kp + "-")[-1]))
                    path.write_text(re.sub(pattern, "", text, flags=re.S), encoding="utf-8")
            caps = json.loads((unit / args.caps).read_text(encoding="utf-8"))
            translations = dict(pair.split("=", 1) for pair in args.en_before.split(";") if pair)
            for caption in caps:
                caption.setdefault("before_en", translations.get(caption["before"], caption["before"]))
            cap_path = Path(temporary) / "captions.json"
            cap_path.write_text(json.dumps(caps, ensure_ascii=False), encoding="utf-8")
            run(repo, sys.executable, tools / "insert_figs.py", cap_path, zh_md, en_md, "--prefix", prefix)
            for name in removed:
                path = repo / "assets/figures" / name
                if path.exists():
                    run(repo, "git", "rm", "--", path)
        run(repo, sys.executable, repo / "scripts/check_slides.py",
            repo / f"slides/{stem}.jbd.json", repo / f"slides/{stem}-en.jbd.json")
        run(repo, sys.executable, repo / "scripts/build_docs.py")
        run(repo, sys.executable, "-m", "mkdocs", "build")
        examples = tools.parent / "examples"
        archived = []
        for pattern in (f"{kp}_nums.py", f"{kp}_pages.py", f"{kp}_text.py", f"{kp}_data.py"):
            for source in unit.glob(pattern):
                destination = examples / source.name
                shutil.copy(source, destination)
                archived.append(destination)
        for source_name, suffix in ((args.caps, "caps.json"), (args.map, "map.json")):
            destination = examples / f"{kp}_{suffix}"
            if (unit / source_name).is_file():
                shutil.copy(unit / source_name, destination)
                archived.append(destination)
        if args.changelog:
            path = repo / "CHANGELOG.md"
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
            position = next((i + 1 for i, line in enumerate(lines) if line.startswith("## ")), 1)
            lines.insert(position, f"\n- {now} — {args.changelog}\n")
            path.write_text("".join(lines), encoding="utf-8")
        if args.state:
            with (repo / "COURSE_STATE.md").open("a", encoding="utf-8") as stream:
                stream.write(f"\n- {now} — {args.state}\n")
        if log:
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", encoding="utf-8") as stream:
                stream.write(f"- {now} — {args.log or args.state or stem}\n")
        run(repo, "git", "add", "--", "CHANGELOG.md", "COURSE_STATE.md",
            "assets/figures", "lectures", "slides", ".agents/skills/view-from-text-v3/examples")
        if log and log.is_relative_to(repo):
            run(repo, "git", "add", "--", log)
        if run(repo, "git", "diff", "--cached", "--quiet", check=False).returncode:
            run(repo, "git", "commit", "-m", f"{kp.upper()} 簡報以 v3 發佈，講義換圖")
            run(repo, "git", "push", "origin", "main")
        for target in others:
            with course_lock(target):
                if run(target, "git", "status", "--porcelain").stdout.strip():
                    raise RuntimeError(f"Sync target changed during publication: {target}")
                run(target, "git", "pull", "--rebase", "origin", "main")
                for source in archived:
                    destination = target / ".agents/skills/view-from-text-v3/examples" / source.name
                    shutil.copy(source, destination)
                    run(target, "git", "add", "--", destination)
                if run(target, "git", "diff", "--cached", "--quiet", check=False).returncode:
                    run(target, "git", "commit", "-m", f"同步 {kp.upper()} v3 範例")
                    run(target, "git", "push", "origin", "main")
    print("PUBLISHED", repo.name, stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
