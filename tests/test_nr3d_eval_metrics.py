import unittest

from eval_helpers import construct_bbox_corners, _compute_grounding_metrics


class Nr3dEvalMetricsTest(unittest.TestCase):
    def test_language_bucket_metrics_and_annotation_flags_are_saved(self) -> None:
        gt_bbox = construct_bbox_corners([0.0, 0.0, 0.0], [2.0, 2.0, 2.0])
        gt_corners_map = {
            "24": gt_bbox,
            "25": gt_bbox,
        }
        annotations = [
            {
                "assignmentid": 101,
                "csv_row_id": 1,
                "scene_id": "scene0011_00",
                "target_id": "24",
                "split": "all",
                "uses_spatial_lang": True,
                "uses_color_lang": False,
                "uses_shape_lang": True,
                "mentions_target_class": True,
            },
            {
                "assignmentid": 202,
                "csv_row_id": 2,
                "scene_id": "scene0011_00",
                "target_id": "25",
                "split": "all",
                "uses_spatial_lang": False,
                "uses_color_lang": True,
                "uses_shape_lang": False,
                "mentions_target_class": False,
            },
        ]
        results = [
            {
                "assignmentid": 101,
                "csv_row_id": 1,
                "ground_truth_target_id": "24",
                "predicted_object_id": "obj_24",
                "bbox_3d": gt_bbox.tolist(),
            },
            {
                "assignmentid": 202,
                "csv_row_id": 2,
                "ground_truth_target_id": "25",
                "predicted_object_id": None,
                "bbox_3d": None,
            },
        ]

        enriched_results, metrics, failures = _compute_grounding_metrics(
            results,
            annotations,
            gt_corners_map,
            (0.1,),
        )

        self.assertEqual(len(enriched_results), 2)
        self.assertEqual(len(failures), 1)
        self.assertTrue(enriched_results[0]["uses_spatial_lang"])
        self.assertFalse(enriched_results[0]["uses_color_lang"])
        self.assertTrue(enriched_results[0]["uses_shape_lang"])
        self.assertTrue(enriched_results[0]["mentions_target_class"])
        self.assertFalse(enriched_results[1]["uses_spatial_lang"])
        self.assertTrue(enriched_results[1]["uses_color_lang"])
        self.assertFalse(enriched_results[1]["uses_shape_lang"])
        self.assertFalse(enriched_results[1]["mentions_target_class"])

        self.assertEqual(metrics["splits"]["overall"]["acc@0.1"], 0.5)
        self.assertEqual(metrics["splits"]["with_spatial_lang"]["acc@0.1"], 1.0)
        self.assertEqual(metrics["splits"]["without_spatial_lang"]["acc@0.1"], 0.0)
        self.assertEqual(metrics["splits"]["with_color_lang"]["acc@0.1"], 0.0)
        self.assertEqual(metrics["splits"]["without_color_lang"]["acc@0.1"], 1.0)
        self.assertEqual(metrics["splits"]["with_shape_lang"]["acc@0.1"], 1.0)
        self.assertEqual(metrics["splits"]["without_shape_lang"]["acc@0.1"], 0.0)
        self.assertEqual(metrics["splits"]["with_target_mention"]["acc@0.1"], 1.0)
        self.assertEqual(metrics["splits"]["without_target_mention"]["acc@0.1"], 0.0)


if __name__ == "__main__":
    unittest.main()
