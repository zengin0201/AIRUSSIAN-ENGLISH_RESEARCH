"""
Run every test plus the task validator. No pytest, no GPU, ~2 seconds.

    python -m tests.run_all

Run this before every experiment, and always after touching a task file.
"""

from __future__ import annotations

import pathlib
import sys
import unittest


def main() -> int:
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py", top_level_dir=".")
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    print("\n" + "=" * 70)
    print("validating task files")
    print("=" * 70)

    from src.validate_tasks import main as validate

    # Every task file, found by glob rather than listed by hand. The list used
    # to be hardcoded, and three probe files added later - including the main
    # one the paper rests on - were silently never validated.
    bad = 0
    paths = sorted(str(p) for p in pathlib.Path("data").glob("tasks_*.json"))
    if not paths:
        print("data/: no task files found")
    for path in paths:
        try:
            bad += validate(["--tasks", path, "--quiet"])
        except FileNotFoundError:
            print(f"{path}: not found, skipped")
        print()

    ok = result.wasSuccessful() and bad == 0
    print("ALL GREEN" if ok else "SOMETHING IS BROKEN — do not run the GPU yet")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
