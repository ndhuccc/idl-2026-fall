"""Write <kpNN>_specs.py for gen_figs.py from a unit's pages module.
    python3 make_specs.py <unit_dir> kp15 <n_pages>
The specs module exposes p1..pN, each returning the converted native figure of that page (light theme PNG for the lecture)."""
import sys, os
d, kp, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
open(os.path.join(d, f"{kp}_specs.py"), "w").write(f'''import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); SK = "/home/user/pr-2026-fall/.agents/skills"
for p in ("view-from-text-v3/tools", "view-from-text-v2/tools", "view-from-text-v2/examples"): sys.path.insert(0, os.path.join(SK, p))
sys.path.insert(0, HERE)
import {kp}_pages as KP
from htmlkit import native
JD = os.path.join(HERE, "out")
def _f(n): return lambda: native(f"{kp}_p{{n}}", f"lec05-{kp}-p{{n}}", KP.W, KP.H, JD, "figure")
{", ".join(f"p{i}" for i in range(1, n + 1))} = (_f(i) for i in range(1, {n + 1}))
''')
print("wrote", f"{kp}_specs.py")
