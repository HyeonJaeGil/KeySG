import unittest

from eval_helpers import construct_bbox_corners

from keysg.utils.iou_eval import strict_box3d_iou


class StrictIoUTest(unittest.TestCase):
    def test_strict_iou_penalizes_containing_box(self) -> None:
        gt_bbox = construct_bbox_corners([0, 0, 0], [1, 1, 1])
        pred_bbox = construct_bbox_corners([0, 0, 0], [2, 2, 2])

        iou = strict_box3d_iou(pred_bbox, gt_bbox)

        self.assertAlmostEqual(iou, 0.125, places=6)


if __name__ == "__main__":
    unittest.main()
