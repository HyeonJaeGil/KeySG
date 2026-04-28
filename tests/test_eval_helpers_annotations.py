import csv
import json
import os
import pathlib
import tempfile
import unittest

from eval_helpers import (
    _eval_output_paths,
    _load_scene_annotations,
    _normalize_annotation,
    _resolve_nr3d_root,
)


class SceneSpecificAnnotationsTest(unittest.TestCase):
    def test_normalize_annotation_uses_assignmentid_and_renames_legacy_ann_id(self) -> None:
        normalized = _normalize_annotation(
            {
                "scan_id": "scene0011_00",
                "utterance": "from scene file",
                "target_id": "2",
                "assignmentid": "41359",
                "ann_id": 40567,
            },
            fallback_id=17,
        )

        assert normalized is not None
        self.assertEqual(normalized["assignmentid"], 41359)
        self.assertEqual(normalized["csv_row_id"], 40567)
        self.assertNotIn("ann_id", normalized)

    def test_scene_file_can_recover_assignmentid_from_shared_csv(self) -> None:
        scene_name = "scene0011_00"

        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            queries_dir = root / "queries_by_scene"
            queries_dir.mkdir()

            with open(root / "nr3d.csv", "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["assignmentid", "scan_id", "utterance", "target_id"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "assignmentid": "9001",
                        "scan_id": scene_name,
                        "utterance": "from scene file",
                        "target_id": "2",
                    }
                )

            with open(queries_dir / f"{scene_name}.json", "w", encoding="utf-8") as handle:
                json.dump(
                    [
                        {
                            "scan_id": scene_name,
                            "utterance": "from scene file",
                            "target_id": "2",
                            "ann_id": 0,
                        }
                    ],
                    handle,
                )

            annotations = _load_scene_annotations(
                f"/unused/path/{scene_name}",
                nr3d_root=str(root),
            )

        self.assertEqual(annotations[0]["assignmentid"], 9001)
        self.assertEqual(annotations[0]["csv_row_id"], 0)

    def test_prefers_unfiltered_scene_file_by_default(self) -> None:
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
                            "assignmentid": 101,
                            "csv_row_id": 7,
                        }
                    ],
                    handle,
                )

            annotations = _load_scene_annotations(
                f"/unused/path/{scene_name}",
                nr3d_root=str(root),
            )

        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0]["utterance"], "from unfiltered scene file")
        self.assertEqual(annotations[0]["target_id"], "1")

    def test_prefers_filtered_scene_file_when_requested(self) -> None:
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
                            "assignmentid": 101,
                            "csv_row_id": 7,
                        }
                    ],
                    handle,
                )

            annotations = _load_scene_annotations(
                f"/unused/path/{scene_name}",
                nr3d_root=str(root),
                use_filtered_queries=True,
            )

        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0]["utterance"], "from filtered scene file")
        self.assertEqual(annotations[0]["target_id"], "2")
        self.assertEqual(annotations[0]["assignmentid"], 101)
        self.assertEqual(annotations[0]["csv_row_id"], 7)

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
                            "assignmentid": 99,
                            "csv_row_id": 4,
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
        self.assertEqual(annotations[0]["assignmentid"], 99)
        self.assertEqual(annotations[0]["csv_row_id"], 4)


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
        self.assertEqual(
            paths["grounding_llm_log"],
            "/tmp/nr3d_eval/scene0011_00_with_frames_grounding_llm.log",
        )
        self.assertEqual(
            paths["analysis_llm_log"],
            "/tmp/nr3d_eval/scene0011_00_with_frames_analysis_llm.log",
        )


if __name__ == "__main__":
    unittest.main()
