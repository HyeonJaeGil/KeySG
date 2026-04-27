# ScanNet Batch Build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Python helper script that runs `keysg-build` for multiple explicit ScanNet scene IDs and supports `--dry-run` verification.

**Architecture:** Keep the existing Hydra pipeline unchanged and add a thin wrapper in `scripts/` that builds per-scene commands and executes them sequentially. Test the wrapper through its CLI surface so the command format stays stable.

**Tech Stack:** Python standard library (`argparse`, `pathlib`, `subprocess`), `unittest`, existing repo `scripts/` conventions

---

### Task 1: Add the CLI regression test

**Files:**
- Create: `tests/test_build_scannet_batch.py`
- Test: `tests/test_build_scannet_batch.py`

- [ ] **Step 1: Write the failing test**

```python
import pathlib
import subprocess
import sys
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_scannet_batch.py"


class BuildScanNetBatchCliTest(unittest.TestCase):
    def test_dry_run_prints_expected_keysg_build_commands(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--scans-root",
                "/data/ScanNet/scans",
                "--output-dir",
                "output/keysg",
                "--scene-id",
                "scene0011_00",
                "--scene-id",
                "scene0022_00",
                "--dry-run",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn(
            "keysg-build dataset.kind=scannet dataset.root_dir=/data/ScanNet/scans/scene0011_00 output_dir=output/keysg",
            result.stdout,
        )
        self.assertIn(
            "keysg-build dataset.kind=scannet dataset.root_dir=/data/ScanNet/scans/scene0022_00 output_dir=output/keysg",
            result.stdout,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_build_scannet_batch.py -v`
Expected: FAIL because `scripts/build_scannet_batch.py` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import shlex
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scans-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scene-id", action="append", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scans_root = pathlib.Path(args.scans_root)

    for scene_id in args.scene_id:
        command = [
            "keysg-build",
            "dataset.kind=scannet",
            f"dataset.root_dir={scans_root / scene_id}",
            f"output_dir={args.output_dir}",
        ]
        if args.dry_run:
            print(" ".join(shlex.quote(part) for part in command))
            continue
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_build_scannet_batch.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_build_scannet_batch.py scripts/build_scannet_batch.py
git commit -m "feat: add ScanNet batch build helper"
```
