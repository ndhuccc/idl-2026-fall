import sys, os, importlib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from lec01_kit import run
mod, out, kz, ke, stem = sys.argv[1:6]
run(importlib.import_module(mod).make, out, kz, ke, stem)
