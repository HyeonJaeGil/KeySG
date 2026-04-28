import io
import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_helpers import _write_debug_entry, _write_llm_entry


class Nr3dEvalDebugLogTest(unittest.TestCase):
    def test_debug_log_includes_readable_payload_summary(self) -> None:
        debug_file = io.StringIO()
        ann = {"assignmentid": 5821, "csv_row_id": 3424, "target_id": "15"}
        selection = SimpleNamespace(confidence=0.9, reason="best match")
        gt_corners_map = {"15": np.zeros((8, 3), dtype=float)}
        gt_label_map = {"15": "trash can"}
        frame_hit = SimpleNamespace(chunk=SimpleNamespace(id="frame_0_0_1236"))
        image_paths = [
            "/tmp/frame_000123.jpg",
            "/tmp/frame_000456.jpg",
        ]

        _write_debug_entry(
            debug_file,
            query_idx=3,
            ann=ann,
            utterance="This trash can is next to a large black television.",
            context_text="USER QUERY: ...",
            selection=selection,
            pred_id="obj_1_1b2",
            bbox=np.zeros((8, 3), dtype=float),
            gt_corners_map=gt_corners_map,
            gt_label_map=gt_label_map,
            frame_results={"frame_visual": [frame_hit]},
            images=[object(), object()],
            payload_summary={
                "model": "gpt-5-mini",
                "reasoning_effort": "medium",
                "detail": "high",
                "response_model": "ObjectSelection",
                "instructions": "System instructions here.",
                "image_paths": image_paths,
            },
        )

        log_text = debug_file.getvalue()

        self.assertIn("payload:", log_text)
        self.assertIn("model: gpt-5-mini", log_text)
        self.assertIn("reasoning_effort: medium", log_text)
        self.assertIn("detail: high", log_text)
        self.assertIn("response_model: ObjectSelection", log_text)
        self.assertIn("instructions:", log_text)
        self.assertIn("System instructions here.", log_text)
        self.assertIn("image_paths:", log_text)
        self.assertIn("/tmp/frame_000123.jpg", log_text)
        self.assertIn("/tmp/frame_000456.jpg", log_text)

    def test_llm_log_includes_payload_and_structured_response(self) -> None:
        llm_file = io.StringIO()
        selection = SimpleNamespace(
            object_id="obj_1_1b2",
            reason="best grounded match",
            confidence=0.9,
            rejected_ids=["obj_20_1f9"],
            guess_id=None,
            model_dump=lambda: {
                "object_id": "obj_1_1b2",
                "reason": "best grounded match",
                "confidence": 0.9,
                "rejected_ids": ["obj_20_1f9"],
                "guess_id": None,
            },
        )

        _write_llm_entry(
            llm_file,
            query_idx=3,
            assignmentid=5821,
            csv_row_id=3424,
            utterance="This trash can is next to a large black television.",
            payload_summary={
                "model": "gpt-5-mini",
                "reasoning_effort": "medium",
                "detail": "high",
                "response_model": "ObjectSelection",
                "instructions": "System instructions here.",
                "image_paths": ["/tmp/frame_000123.jpg"],
                "context_text": "USER QUERY: ...",
            },
            selection=selection,
        )

        log_text = llm_file.getvalue()

        self.assertIn("## Final Grounding Query 3", log_text)
        self.assertIn("assignmentid: 5821", log_text)
        self.assertIn("csv_row_id: 3424", log_text)
        self.assertIn("payload:", log_text)
        self.assertIn("context_text:", log_text)
        self.assertIn("response:", log_text)
        self.assertIn('"object_id": "obj_1_1b2"', log_text)
        self.assertIn('"confidence": 0.9', log_text)

    def test_llm_log_can_write_query_analysis_entries(self) -> None:
        llm_file = io.StringIO()
        analysis = SimpleNamespace(
            model_dump=lambda: {
                "target_object": "trash can",
                "anchor_objects": ["television"],
                "relation_polarity": "near",
            }
        )

        _write_llm_entry(
            llm_file,
            query_idx=3,
            assignmentid=5821,
            csv_row_id=3424,
            utterance="This trash can is next to a large black television.",
            payload_summary={
                "model": "gpt-5-nano",
                "reasoning_effort": "low",
                "detail": "auto",
                "response_model": "_QuerySchema",
                "instructions": "Analysis instructions here.",
                "image_paths": [],
                "context_text": "User query: This trash can is next to a large black television.",
            },
            selection=analysis,
            entry_title="Query Analysis",
        )

        log_text = llm_file.getvalue()

        self.assertIn("## Query Analysis 3", log_text)
        self.assertIn("Analysis instructions here.", log_text)
        self.assertIn('"target_object": "trash can"', log_text)
        self.assertIn('"anchor_objects": [', log_text)


if __name__ == "__main__":
    unittest.main()
