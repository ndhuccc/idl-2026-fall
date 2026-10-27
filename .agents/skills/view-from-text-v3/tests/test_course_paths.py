"""Run with: python3 -m unittest discover -s .agents/skills/view-from-text-v3/tests -v"""
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SKILL = Path(__file__).resolve().parents[1]
TOOLS = SKILL / "tools"


class CoursePaths(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="v3-course-test-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.env = {k: v for k, v in os.environ.items()
                    if k not in ("COURSE_REPO", "COURSE_SKILLS", "SKILLS_DIR", "LEC", "PYTHONPATH")}
        self.env["PYTHONDONTWRITEBYTECODE"] = "1"

    def course(self, name, title):
        root = self.base / name
        tools = root / ".agents/skills/view-from-text-v3/tools"
        tools.mkdir(parents=True)
        for filename in ("course_context.py", "publish_unit.py"):
            shutil.copy(TOOLS / filename, tools)
        common = root / ".agents/skills/view-from-text-v2/tools/deckcommon.py"
        common.parent.mkdir(parents=True)
        common.write_text("# fixture\n", encoding="utf-8")
        (root / "COURSE_STATE.md").write_text(f"| 課程 | {title}(TEST) |\n", encoding="utf-8")
        return root

    def execute(self, script, *args, env=None, check=True):
        return subprocess.run([sys.executable, str(script), *map(str, args)],
                              cwd=self.base, env=env or self.env,
                              text=True, capture_output=True, check=check)

    def inspect(self, directory, module, expression, env=None):
        code = ("import sys,runpy,json;sys.path.insert(0,sys.argv[1]);"
                "d=runpy.run_path(sys.argv[1]+'/'+sys.argv[2]);"
                f"print(json.dumps({expression},ensure_ascii=False))")
        result = self.execute("-c", code, directory, module, env=env)
        return json.loads(result.stdout)

    def scaffold(self, root, name="unit", lecture="lec08"):
        directory = self.base / name
        self.execute(TOOLS / "new_unit.py", directory, "KP03", "單元", "Unit",
                     "--repo", root, "--lecture", lecture)
        return directory

    def test_scaffold_selects_each_course_and_chapter(self):
        for name, title, lecture in (("pr course", "圖型識別", "lec08"),
                                     ("idl course", "深度學習導論", "lec02")):
            with self.subTest(course=name):
                root = self.course(name, title)
                unit = self.scaffold(root, name + " unit", lecture)
                config = json.loads((unit / "course-unit.json").read_text(encoding="utf-8"))
                self.assertEqual(config["repo_root"], str(root))
                self.assertEqual(config["lecture"], lecture)
                title_zh, title_en = self.inspect(unit, "kp03_text.py", "[d['TITLE_ZH'],d['TITLE_EN']]")
                self.assertIn(title, title_zh)
                self.assertEqual(title_en, "KP03: Unit")
                for path in unit.glob("*.py"):
                    ast.parse(path.read_text(encoding="utf-8"))

    def test_generated_module_in_language_work_folder(self):
        root = self.course("idl", "深度學習導論")
        unit = self.scaffold(root)
        work = unit / "W_en"
        for name in ("course_context.py", "kp03_text.py"):
            shutil.copy(unit / name, work)
        context = self.inspect(work, "kp03_text.py", "d['CONTEXT']")
        self.assertEqual(context["repo_root"], str(root))
        self.assertEqual(context["lecture"], "lec08")

    def test_relocated_course_can_be_selected_explicitly(self):
        root = self.course("original", "深度學習導論")
        unit = self.scaffold(root)
        moved = self.base / "moved course"
        root.rename(moved)
        env = dict(self.env, COURSE_REPO=str(moved))
        context = self.inspect(unit, "kp03_text.py", "d['CONTEXT']", env=env)
        self.assertEqual(context["repo_root"], str(moved))

    def test_single_page_specs_use_selected_lecture(self):
        root = self.course("idl", "深度學習導論")
        unit = self.scaffold(root)
        (unit / "kp03_pages.py").write_text("W,H=1160,470\n", encoding="utf-8")
        (unit / "htmlkit.py").write_text("def native(*args): return args\n", encoding="utf-8")
        self.execute(TOOLS / "make_specs.py", unit, "kp03", "1", "--repo", root)
        values = self.inspect(unit, "kp03_specs.py", "[callable(d['p1']),d['p1']()[1]]")
        self.assertEqual(values, [True, "lec08-kp03-p1"])

    def test_missing_legacy_lecture_does_not_default_to_chapter_five(self):
        root = self.course("idl", "深度學習導論")
        unit = self.base / "legacy"
        unit.mkdir()
        result = self.execute(TOOLS / "make_specs.py", unit, "kp03", "1",
                              "--repo", root, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--lecture is required", result.stderr)
        self.assertFalse((unit / "kp03_specs.py").exists())

    def test_dry_run_defaults_to_tool_course_without_mutation_or_git(self):
        for name in ("pr", "idl"):
            with self.subTest(course=name):
                root = self.course(name, name)
                script = root / ".agents/skills/view-from-text-v3/tools/publish_unit.py"
                before = {str(p): p.read_bytes() for p in self.base.rglob("*") if p.is_file()}
                env = dict(self.env, PATH="")
                result = self.execute(script, "--unit-dir", self.base / "not-created",
                                      "--kp", "kp03", "--pages", "1",
                                      "--lecture", "lec08-kp03-topic",
                                      "--map", "map.json", "--caps", "caps.json",
                                      "--dry-run", env=env)
                plan = json.loads(result.stdout)
                self.assertEqual(plan["repo"], str(root))
                self.assertEqual(plan["prefix"], "lec08-kp03")
                self.assertEqual(plan["sync_to"], [])
                self.assertIsNone(plan["log_file"])
                after = {str(p): p.read_bytes() for p in self.base.rglob("*") if p.is_file()}
                self.assertEqual(before, after)

    def test_explicit_sync_and_log_appear_only_in_plan(self):
        root = self.course("pr", "圖型識別")
        other = self.course("idl", "深度學習導論")
        result = self.execute(TOOLS / "publish_unit.py", "--repo", root,
                              "--unit-dir", self.base / "unit", "--kp", "kp03",
                              "--pages", "1", "--lecture", "lec08-kp03-topic",
                              "--map", "map.json", "--caps", "caps.json",
                              "--sync-to", other, "--log-file", "logs/units.md", "--dry-run")
        plan = json.loads(result.stdout)
        self.assertEqual(plan["sync_to"], [str(other)])
        self.assertEqual(plan["log_file"], str(root / "logs/units.md"))
        self.assertFalse((root / "logs").exists())

    def test_mismatched_unit_is_rejected(self):
        root = self.course("idl", "深度學習導論")
        result = self.execute(TOOLS / "publish_unit.py", "--repo", root,
                              "--unit-dir", self.base, "--kp", "kp04", "--pages", "1",
                              "--lecture", "lec08-kp03-topic", "--map", "map.json",
                              "--caps", "caps.json", "--dry-run", check=False)
        self.assertNotEqual(result.returncode, 0)

    def test_caption_prefix_is_inferred_from_lecture(self):
        zh = self.base / "lec08-kp03-topic.md"
        en = self.base / "lec08-kp03-topic-en.md"
        zh.write_text("### 開始\n\n正文\n", encoding="utf-8")
        en.write_text("### Start\n\nText\n", encoding="utf-8")
        caps = self.base / "caps.json"
        caps.write_text(json.dumps([dict(id="overview", before="開始", before_en="Start",
                                        zh=["圖說", "替代"], en=["Caption", "Alt"])]),
                        encoding="utf-8")
        self.execute(TOOLS / "insert_figs.py", caps, zh, en)
        self.assertIn("lec08-kp03-overview.png", zh.read_text(encoding="utf-8"))
        self.assertIn("lec08-kp03-overview-en.png", en.read_text(encoding="utf-8"))

    def test_publication_follows_single_or_split_lecture_layout(self):
        for name, split in (("idl", False), ("pr", True)):
            with self.subTest(course=name):
                root = self.course(name, name)
                lecture = root / "lectures/lec08-kp03-topic.md"
                lecture.parent.mkdir()
                lecture.write_text("Approved bilingual lecture\n", encoding="utf-8")
                if split:
                    english = root / "lectures/en/lec08-kp03-topic.md"
                    english.parent.mkdir()
                    english.write_text("Approved English lecture\n", encoding="utf-8")
                result = self.execute(TOOLS / "publish_unit.py", "--repo", root,
                                      "--unit-dir", self.base / "not-created", "--kp", "kp03",
                                      "--pages", "1", "--lecture", "lec08-kp03-topic", "--dry-run")
                plan = json.loads(result.stdout)
                self.assertEqual(plan["lecture_layout"], "split" if split else "bilingual")
                self.assertEqual(plan["lecture_figures"], "regenerate" if split else "preserve")
                self.assertEqual(lecture.read_text(encoding="utf-8"), "Approved bilingual lecture\n")

    def test_bilingual_lecture_cannot_accidentally_replace_figures(self):
        root = self.course("idl", "深度學習導論")
        result = self.execute(TOOLS / "publish_unit.py", "--repo", root,
                              "--unit-dir", self.base, "--kp", "kp03", "--pages", "1",
                              "--lecture", "lec08-kp03-topic", "--lecture-figures", "regenerate",
                              "--dry-run", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("keeps approved figures", result.stderr)

    def test_python_sources_parse_and_runtime_paths_are_portable(self):
        paths = list(SKILL.rglob("*.py"))
        paths += [SKILL.parent / "view-from-text-v2/tools/deckcommon.py",
                  SKILL.parent / "view-from-text-v2/examples/kp18_pnn_html.py"]
        for path in paths:
            with self.subTest(path=path.name):
                source = path.read_text(encoding="utf-8")
                ast.parse(source, filename=str(path))
                self.assertNotIn("/home/" + "user/pr-2026-fall", source)
                self.assertNotIn("/tmp/" + "claude-", source)
                self.assertNotIn("/workspace/" + "idl-2026-fall", source)


if __name__ == "__main__":
    unittest.main()
