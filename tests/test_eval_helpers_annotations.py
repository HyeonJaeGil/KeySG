import csv
import json
import pathlib
import tempfile
import unittest

from eval_helpers import _load_scene_annotations


class SceneSpecificAnnotationsTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
