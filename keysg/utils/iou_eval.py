from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


def _construct_bbox_corners(
    bbox_center: np.ndarray, bbox_extent: np.ndarray
) -> np.ndarray:
    half = bbox_extent / 2.0
    offsets = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    )
    return bbox_center + offsets * half


def bbox_to_corners(bbox: Any) -> Optional[np.ndarray]:
    if bbox is None:
        return None

    arr = np.asarray(bbox, dtype=float)
    if arr.size == 0:
        return None
    if arr.shape == (8, 3):
        return arr
    if arr.shape == (2, 3):
        lo, hi = arr
        return _construct_bbox_corners((lo + hi) / 2.0, hi - lo)
    if arr.shape == (6,):
        return _construct_bbox_corners(arr[:3], arr[3:])
    if arr.shape == (3, 2):
        lo = arr[:, 0]
        hi = arr[:, 1]
        return _construct_bbox_corners((lo + hi) / 2.0, hi - lo)
    raise ValueError(f"Unsupported bbox shape: {arr.shape}")


def axis_aligned_bounds_from_corners(
    bbox: Any,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    corners = bbox_to_corners(bbox)
    if corners is None:
        return None
    return corners.min(axis=0), corners.max(axis=0)


def strict_box3d_iou(pred_bbox: Any, gt_bbox: Any) -> float:
    pred_bounds = axis_aligned_bounds_from_corners(pred_bbox)
    gt_bounds = axis_aligned_bounds_from_corners(gt_bbox)
    if pred_bounds is None or gt_bounds is None:
        return 0.0

    pred_min, pred_max = pred_bounds
    gt_min, gt_max = gt_bounds

    inter_min = np.maximum(pred_min, gt_min)
    inter_max = np.minimum(pred_max, gt_max)
    inter_size = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = float(np.prod(inter_size))
    if inter_vol <= 0.0:
        return 0.0

    pred_vol = float(np.prod(np.maximum(pred_max - pred_min, 0.0)))
    gt_vol = float(np.prod(np.maximum(gt_max - gt_min, 0.0)))
    if pred_vol <= 0.0 or gt_vol <= 0.0:
        return 0.0

    union_vol = pred_vol + gt_vol - inter_vol
    if union_vol <= 0.0:
        return 0.0

    return inter_vol / union_vol
