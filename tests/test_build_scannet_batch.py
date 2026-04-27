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


if __name__ == "__main__":
    unittest.main()
