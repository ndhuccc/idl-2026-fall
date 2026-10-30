"""Run with python3 -m unittest discover -s tests -v."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from jbd_page import Page
from new_deck import make


class GeometryTests(unittest.TestCase):
    def test_contain(self):
        for w, h in [(1600, 900), (1000, 1000), (600, 1200), (2000, 500)]:
            with self.subTest(size=(w, h)):
                p = Page(w, h)
                n = p.rect(0, 0, w, h)
                k = min(1280 / w, 720 / h)
                self.assertAlmostEqual(n["width"], w * k, places=2)
                self.assertAlmostEqual(n["height"], h * k, places=2)
                self.assertAlmostEqual(n["transform"]["x"], (1280 - w * k) / 2, places=2)
                self.assertAlmostEqual(n["transform"]["y"], (720 - h * k) / 2, places=2)
                self.assertEqual(p.check(), [])
    def test_invalid_dimensions(self):
        for size in [(0, 100), (-1, 100), (100, float("nan"))]:
            with self.assertRaises(ValueError):
                Page(*size)


class InsertionTests(unittest.TestCase):
    def test_repeated_insert_and_replace_preserve_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            deck, nodes = root / "deck.json", root / "nodes.json"
            deck.write_text(json.dumps(make()))
            p = Page(); p.text(20, 30, 300, 40, "Original"); p.dump(nodes)
            def run(*args):
                return subprocess.run([sys.executable, str(TOOLS / "insert_page.py"),
                                       str(deck), "--nodes", str(nodes), *args],
                                      capture_output=True, text=True)
            self.assertEqual(run("--after", "0").returncode, 0)
            original = json.loads(deck.read_text())
            first_id = original["scenes"]["scene-u01"]["nodeIds"][0]
            for _ in range(3):
                self.assertEqual(run("--after", "0").returncode, 0)
            data = json.loads(deck.read_text())
            self.assertEqual(len(data["scenes"]["scene-u01"]["nodeIds"]), 4)
            self.assertEqual(data["nodes"][first_id]["text"], "Original")
            self.assertEqual(data["nodes"][first_id]["transform"]["x"], 20 + 3 * 1440)
            self.assertEqual([s["target"]["point"]["x"] for s in data["steps"][1:]],
                             [640 + i * 1440 for i in range(4)])
            untouched = {k:v for k,v in data["nodes"].items()
                         if k not in data["scenes"]["scene-u01"]["nodeIds"] or
                         v["transform"]["x"] >= 1440}
            for _ in range(2):
                self.assertEqual(run("--replace", "1").returncode, 0)
            final = json.loads(deck.read_text())
            self.assertEqual(len(final["scenes"]["scene-u01"]["nodeIds"]), 4)
            for k,v in untouched.items():
                self.assertEqual(final["nodes"][k], v)
            before = deck.read_bytes()
            collision_prefix = first_id.rsplit("-", 1)[0]
            result = run("--after", "0", "--prefix", collision_prefix)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(deck.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
