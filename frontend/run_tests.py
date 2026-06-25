#!/usr/bin/env python
"""Dependency-free test runner (pytest is not installed in the spik-yolo env).

Discovers tests/test_*.py, runs every top-level `test_*` function, reports pass/fail, and exits
nonzero on any failure. Run from the frontend/ dir:

  /home/twt/.conda/envs/spik-yolo/bin/python run_tests.py [substring-filter]
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path


def main() -> int:
    root = Path(__file__).parent.resolve()
    sys.path.insert(0, str(root))  # make `import neurort_compiler` work without install
    name_filter = sys.argv[1] if len(sys.argv) > 1 else ""

    passed = 0
    fails = []
    for tf in sorted((root / "tests").glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(tf.stem, tf)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as e:  # import-time failure
            fails.append((tf.stem, "<import>", traceback.format_exc()))
            print(f"FAIL {tf.stem} <import>: {e}")
            continue
        for name in sorted(dir(mod)):
            if not name.startswith("test_") or not callable(getattr(mod, name)):
                continue
            if name_filter and name_filter not in f"{tf.stem}::{name}":
                continue
            try:
                getattr(mod, name)()
                passed += 1
                print(f"PASS {tf.stem}::{name}")
            except Exception as e:
                fails.append((tf.stem, name, traceback.format_exc()))
                print(f"FAIL {tf.stem}::{name}: {e}")

    print(f"\n{passed} passed, {len(fails)} failed")
    for stem, name, tb in fails:
        print(f"\n--- {stem}::{name} ---\n{tb}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
