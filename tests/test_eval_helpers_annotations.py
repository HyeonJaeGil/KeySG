import csv
import json
import os
import pathlib
import tempfile
import unittest

from eval_helpers import (
    _eval_output_paths,
    _load_scene_annotations,
    _resolve_nr3d_root,
)


class SceneSpecificAnnotationsTest(unittest.TestCase):
    def test_prefers_filtered_scene_file_over_unfiltered_scene_file(self) -> None:
        scene_name = "scene0011_00"

        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            filtered_dir = root / "queries_by_scene_filtered"
            unfiltered_dir = root / "queries_by_scene"
            filtered_dir.mkdir()
            unfiltered_dir.mkdir()

            with open(
                unfiltered_dir / f"{scene_name}.json", "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    [
                        {
                            "scan_id": scene_name,
                            "utterance": "from unfiltered scene file",
                            "target_id": "1",
                        }
                    ],
                    handle,
                )

            with open(
                filtered_dir / f"{scene_name}.json", "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    [
                        {
                            "scan_id": scene_name,
                            "utterance": "from filtered scene file",
                            "target_id": "2",
                            "ann_id": 101,
                        }
                    ],
                    handle,
                )

            annotations = _load_scene_annotations(
                f"/unused/path/{scene_name}",
                nr3d_root=str(root),
            )

        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0]["utterance"], "from filtered scene file")
        self.assertEqual(annotations[0]["target_id"], "2")
        self.assertEqual(annotations[0]["ann_id"], 101)

    def test_prefers_queries_by_scene_file_over_shared_csv(self) -> None:
        scene_name = "scene0011_00"

        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            queries_dir = root / "queries_by_scene"
            queries_dir.mkdir()

            with open(root / "nr3d.csv", "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["scan_id", "utterance", "target_id"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "scan_id": scene_name,
                        "utterance": "from csv",
                        "target_id": "1",
                    }
                )

            with open(queries_dir / f"{scene_name}.json", "w", encoding="utf-8") as handle:
                json.dump(
                    [
                        {
                            "scan_id": scene_name,
                            "utterance": "from scene file",
                            "target_id": "2",
                            "ann_id": 99,
                        }
                    ],
                    handle,
                )

            annotations = _load_scene_annotations(
                f"/unused/path/{scene_name}",
                nr3d_root=str(root),
            )

        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0]["utterance"], "from scene file")
        self.assertEqual(annotations[0]["target_id"], "2")
        self.assertEqual(annotations[0]["ann_id"], 99)


class Nr3dRootResolutionTest(unittest.TestCase):
    def test_uses_local_nr3d_data_directory_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            nr3d_dir = root / "nr3d_data"
            (nr3d_dir / "queries_by_scene").mkdir(parents=True)
            (nr3d_dir / "gt_bbox").mkdir()

            prev_cwd = os.getcwd()
            try:
                os.chdir(root)
                resolved = _resolve_nr3d_root(None)
            finally:
                os.chdir(prev_cwd)

        self.assertEqual(resolved, str(nr3d_dir))


class EvalOutputPathsTest(unittest.TestCase):
    def test_custom_run_name_changes_all_output_filenames(self) -> None:
        paths = _eval_output_paths(
            output_dir="/tmp/nr3d_eval",
            scene_dir="/tmp/scenes/scene0011_00",
            run_name="with_frames",
        )

        self.assertEqual(
            paths["results"],
            "/tmp/nr3d_eval/scene0011_00_with_frames_results.json",
        )
        self.assertEqual(
            paths["metrics"],
            "/tmp/nr3d_eval/scene0011_00_with_frames_metrics.json",
        )
        self.assertEqual(
            paths["failed"],
            "/tmp/nr3d_eval/scene0011_00_with_frames_failed.json",
        )
        self.assertEqual(
            paths["summary"],
            "/tmp/nr3d_eval/scene0011_00_with_frames_summary.txt",
        )
        self.assertEqual(
            paths["debug"],
            "/tmp/nr3d_eval/scene0011_00_with_frames_debug.log",
        )


if __name__ == "__main__":
    unittest.main()
