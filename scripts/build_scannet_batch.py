#!/usr/bin/env python3
"""Run `keysg-build` sequentially for multiple ScanNet scene IDs."""

from __future__ import annotations

import argparse
import pathlib
import shlex
import subprocess
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run keysg-build for multiple ScanNet scenes."
    )
    parser.add_argument(
        "--scans-root",
        required=True,
        help="Base directory containing ScanNet scene folders.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Base KeySG output directory. The pipeline will append ScanNet/<scene_id>.",
    )
    parser.add_argument(
        "--scene-id",
        nargs="+",
        required=True,
        help="One or more ScanNet scene IDs to build.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args(argv)


def build_command(scans_root: pathlib.Path, scene_id: str, output_dir: str) -> list[str]:
    return [
        "keysg-build",
        "dataset.kind=scannet",
        f"dataset.root_dir={scans_root / scene_id}",
        f"output_dir={output_dir}",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    scans_root = pathlib.Path(args.scans_root)

    for scene_id in args.scene_id:
        command = build_command(scans_root, scene_id, args.output_dir)
        if args.dry_run:
            print(" ".join(shlex.quote(part) for part in command))
            continue
        subprocess.run(command, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
