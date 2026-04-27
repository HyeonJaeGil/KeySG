import importlib.util
import pathlib
import unittest
from types import SimpleNamespace


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "nr3d_eval.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("nr3d_eval_module", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


class GroundingResultRowTest(unittest.TestCase):
    def test_row_keeps_visualization_friendly_retrieval_context(self) -> None:
        module = _load_module()

        ann = {"ann_id": 7, "target_id": "24", "scene_id": "scene0011_00", "split": "all"}
        frame_chunk = SimpleNamespace(
            id="frame_0_0_26",
            content="frame description",
            metadata={
                "room_id": "0_0",
                "frame_index": 26,
                "image_path": "/tmp/frame.jpg",
                "labeled_image_path": "/tmp/frame_labeled.jpg",
            },
        )
        target_hit = SimpleNamespace(
            chunk=SimpleNamespace(
                id="obj_1_bee",
                content="wood-framed window near doors",
                metadata={"room_id": "0_0", "object_id": "obj_1_bee", "name": "window"},
            ),
            score=0.91,
        )
        selection = SimpleNamespace(
            object_id="obj_1_bee",
            reason="best semantic and spatial match",
            confidence=0.82,
            rejected_ids=["obj_3_abc"],
            guess_id=None,
        )

        row = module._build_grounding_result_row(
            scene_id="scene0011_00",
            ann=ann,
            utterance="The window nearest the front doors.",
            parsed_target="window",
            parsed_anchor_objects=["front doors"],
            target_vis=[target_hit],
            anchor_vis=[],
            top_frame_chunks=[frame_chunk],
            spatial_rel_lines=["candidate near front doors"],
            selection=selection,
            pred_id="obj_1_bee",
            pred_label="wood-framed window",
            bbox_3d=[[0, 0, 0]] * 8,
            timestamp="2026-04-27T00:00:00Z",
        )

        self.assertEqual(row["predicted_object_id"], "obj_1_bee")
        self.assertEqual(row["predicted_label"], "wood-framed window")
        self.assertEqual(row["confidence"], 0.82)
        self.assertEqual(row["reason"], "best semantic and spatial match")
        self.assertEqual(row["parsed_target"], "window")
        self.assertEqual(row["parsed_anchor_objects"], ["front doors"])
        self.assertEqual(row["retrieval"]["target_candidates"][0]["id"], "obj_1_bee")
        self.assertEqual(row["retrieval"]["frames"][0]["id"], "frame_0_0_26")
        self.assertEqual(row["retrieval"]["frames"][0]["image_path"], "/tmp/frame.jpg")


if __name__ == "__main__":
    unittest.main()
