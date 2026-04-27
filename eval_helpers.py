from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import shutil
import sys
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, TextIO, Tuple

import numpy as np
from loguru import logger
from keysg.utils.iou_eval import strict_box3d_iou


_EVAL_KEYS = ("num_queries", "num_predictions", "mean_iou")
_ANNOTATION_FLAG_FIELDS = (
    "uses_spatial_lang",
    "uses_color_lang",
    "uses_shape_lang",
    "mentions_target_class",
)
_LANGUAGE_BUCKETS = (
    ("with_spatial_lang", "uses_spatial_lang", True),
    ("without_spatial_lang", "uses_spatial_lang", False),
    ("with_color_lang", "uses_color_lang", True),
    ("without_color_lang", "uses_color_lang", False),
    ("with_shape_lang", "uses_shape_lang", True),
    ("without_shape_lang", "uses_shape_lang", False),
    ("with_target_mention", "mentions_target_class", True),
    ("without_target_mention", "mentions_target_class", False),
)


def _scene_base(scene_dir: str) -> str:
    return os.path.basename(os.path.normpath(scene_dir))


def _resolve_nr3d_root(nr3d_root: Optional[str]) -> str:
    if nr3d_root:
        if os.path.isdir(nr3d_root):
            return nr3d_root
        raise FileNotFoundError(f"NR3D root does not exist: {nr3d_root}")

    repo_root = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(os.path.join(os.getcwd(), "nr3d_data")),
        os.path.join(repo_root, "nr3d_data"),
    ]

    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    raise FileNotFoundError(
        "NR3D root not provided and no local ./nr3d_data directory was found"
    )


def _eval_output_paths(
    output_dir: str,
    scene_dir: str,
    run_name: str,
    script_path: Optional[str] = None,
) -> Dict[str, str]:
    base = _scene_base(scene_dir)
    stem = f"{base}_{run_name}"
    paths = {
        "results": os.path.join(output_dir, f"{stem}_results.json"),
        "metrics": os.path.join(output_dir, f"{stem}_metrics.json"),
        "failed": os.path.join(output_dir, f"{stem}_failed.json"),
        "summary": os.path.join(output_dir, f"{stem}_summary.txt"),
        "debug": os.path.join(output_dir, f"{stem}_debug.log"),
        "args": os.path.join(output_dir, f"{stem}_args.json"),
        "meta": os.path.join(output_dir, f"{stem}_run_meta.json"),
    }
    if script_path:
        script_name = os.path.basename(script_path)
        paths["script_copy"] = os.path.join(output_dir, f"{stem}_{script_name}")
    return paths


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def construct_bbox_corners(
    bbox_center: Sequence[float], bbox_extent: Sequence[float]
) -> np.ndarray:
    center = np.asarray(bbox_center, dtype=float).reshape(3)
    extent = np.asarray(bbox_extent, dtype=float).reshape(3)
    half = extent / 2.0
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
    return center + offsets * half


def _safe_bbox_from_center_extent(
    bbox_center: Optional[Sequence[float]], bbox_extent: Optional[Sequence[float]]
) -> Optional[np.ndarray]:
    if bbox_center is None or bbox_extent is None:
        return None
    try:
        return construct_bbox_corners(bbox_center, bbox_extent)
    except Exception:
        return None


def _bbox_to_np(bbox: Any) -> Optional[np.ndarray]:
    if bbox is None:
        return None
    arr = np.asarray(bbox, dtype=float)
    if arr.size == 0:
        return None
    if arr.shape == (8, 3):
        return arr
    if arr.shape == (2, 3):
        lo, hi = arr
        return construct_bbox_corners((lo + hi) / 2.0, hi - lo)
    if arr.shape == (6,):
        return construct_bbox_corners(arr[:3], arr[3:])
    if arr.shape == (3, 2):
        lo = arr[:, 0]
        hi = arr[:, 1]
        return construct_bbox_corners((lo + hi) / 2.0, hi - lo)
    raise ValueError(f"Unsupported bbox shape: {arr.shape}")


def _extract_bbox_corners(obj: Any) -> Optional[np.ndarray]:
    bbox = None
    if isinstance(obj, dict):
        bbox = obj.get("bbox_3d")
        if bbox is None:
            bbox = _safe_bbox_from_center_extent(
                obj.get("bbox_center"), obj.get("bbox_extent")
            )
    else:
        bbox = getattr(obj, "bbox_3d", None)
    if bbox is not None:
        try:
            return _bbox_to_np(bbox)
        except Exception:
            pass

    pcd = obj.get("pcd") if isinstance(obj, dict) else getattr(obj, "pcd", None)
    if pcd is not None:
        try:
            pts = np.asarray(pcd.points)
            if len(pts) > 0:
                lo = pts.min(axis=0)
                hi = pts.max(axis=0)
                return construct_bbox_corners((lo + hi) / 2.0, hi - lo)
        except Exception:
            pass
    return None


def _bbox_min_max(corners: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    bbox = _bbox_to_np(corners)
    if bbox is None:
        return None
    return bbox.min(axis=0), bbox.max(axis=0)


def box3d_iou(pred_bbox: Any, gt_bbox: Any) -> float:
    pred_mm = _bbox_min_max(pred_bbox)
    gt_mm = _bbox_min_max(gt_bbox)
    if pred_mm is None or gt_mm is None:
        return 0.0

    pred_min, pred_max = pred_mm
    gt_min, gt_max = gt_mm

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

    union = pred_vol + gt_vol - inter_vol
    if union <= 0.0:
        return 0.0

    iou = inter_vol / union
    return max(iou, inter_vol / min(pred_vol, gt_vol))


def _z_score_normalize(values: Sequence[float]) -> List[float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return []
    std = float(arr.std())
    if std < 1e-9:
        return [0.0 for _ in arr]
    mean = float(arr.mean())
    return ((arr - mean) / std).tolist()


def _get_obj_center(obj: Any) -> Optional[np.ndarray]:
    pcd = getattr(obj, "pcd", None)
    if pcd is not None:
        try:
            pts = np.asarray(pcd.points)
            if len(pts) > 0:
                return pts.mean(axis=0)
        except Exception:
            pass

    bbox = _extract_bbox_corners(obj)
    if bbox is None:
        return None
    return bbox.mean(axis=0)


def _compute_scene_center(objects: Sequence[Any]) -> np.ndarray:
    centers = [center for obj in objects if (center := _get_obj_center(obj)) is not None]
    if not centers:
        return np.zeros(3, dtype=float)
    return np.asarray(centers, dtype=float).mean(axis=0)


def _rank_frame_ids(
    frame_results: Dict[str, Sequence[Any]],
    top_k: int,
    include_visual: bool = True,
    include_text: bool = False,
) -> List[str]:
    seen: set[str] = set()
    ranked: List[str] = []
    sources: List[Sequence[Any]] = []
    if include_visual:
        sources.append(frame_results.get("frame_visual", []))
    if include_text:
        sources.append(frame_results.get("text", []))
    for results in sources:
        for result in results:
            chunk_id = result.chunk.id
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            ranked.append(chunk_id)
            if len(ranked) >= top_k:
                return ranked
    return ranked


def _load_frame_images(frame_chunks: List[Any], max_images: int = 4) -> List[Any]:
    from PIL import Image as PILImage
    from PIL import ImageDraw, ImageFont

    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]

    def _font(size: int = 22) -> Any:
        for path in font_paths:
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
        return ImageFont.load_default()

    images: List[Any] = []
    for chunk in frame_chunks[:max_images]:
        meta = getattr(chunk, "metadata", None) or {}
        path = meta.get("labeled_image_path") or meta.get("image_path")
        if not path or not os.path.isfile(path):
            continue
        try:
            image = PILImage.open(path).convert("RGB")
            draw = ImageDraw.Draw(image)
            label = f"FRAME_ID={chunk.id}"
            font = _font(22)
            try:
                box = draw.textbbox((0, 0), label, font=font)
                width, height = box[2] - box[0], box[3] - box[1]
            except AttributeError:
                width, height = len(label) * 13, 22
            pad = 5
            draw.rectangle([0, 0, width + pad * 2, height + pad * 2], fill=(0, 0, 0))
            draw.text((pad, pad), label, fill=(255, 230, 0), font=font)
            images.append(image)
        except Exception as exc:
            logger.debug("Could not load frame image {}: {}", path, exc)
    return images


def _build_spatial_relations(
    target_vis: Sequence[Any],
    anchor_vis: Sequence[Any],
    obj_by_id: Dict[str, Any],
    scene_center: Optional[np.ndarray] = None,
) -> List[str]:
    del scene_center
    lines: List[str] = []
    for target_result in target_vis[:3]:
        target_obj = obj_by_id.get(target_result.chunk.id)
        if target_obj is None:
            continue
        target_pcd = getattr(target_obj, "pcd", None)
        if target_pcd is None or len(target_pcd.points) == 0:
            continue
        target_points = np.asarray(target_pcd.points)
        target_center = target_points.mean(axis=0)
        target_min = np.asarray(target_pcd.get_min_bound())
        target_max = np.asarray(target_pcd.get_max_bound())

        for anchor_result in anchor_vis[:3]:
            anchor_obj = obj_by_id.get(anchor_result.chunk.id)
            if anchor_obj is None:
                continue
            anchor_pcd = getattr(anchor_obj, "pcd", None)
            if anchor_pcd is None or len(anchor_pcd.points) == 0:
                continue
            anchor_points = np.asarray(anchor_pcd.points)
            anchor_center = anchor_points.mean(axis=0)
            anchor_min = np.asarray(anchor_pcd.get_min_bound())
            anchor_max = np.asarray(anchor_pcd.get_max_bound())

            delta = target_center - anchor_center
            distance = float(np.linalg.norm(delta))
            horizontal_distance = float(np.linalg.norm(delta[[0, 2]]))

            directions: List[str] = []
            if abs(delta[0]) > 0.2:
                directions.append(
                    "to the right of" if delta[0] > 0 else "to the left of"
                )
            if abs(delta[1]) > 0.2:
                directions.append("above" if delta[1] > 0 else "below")
            if abs(delta[2]) > 0.2:
                directions.append("in front of" if delta[2] > 0 else "behind")

            target_y_extent = float(target_max[1] - target_min[1])
            anchor_y_extent = float(anchor_max[1] - anchor_min[1])
            vertical_gap = abs(delta[1])
            if vertical_gap < (target_y_extent + anchor_y_extent) * 0.3 and delta[1] > 0:
                directions.append("on top of")

            relation = ", ".join(directions) if directions else "near"
            lines.append(
                f"ID={target_result.chunk.id} ({getattr(target_obj, 'label', '?')}) "
                f"is {distance:.2f}m ({relation}) "
                f"ID={anchor_result.chunk.id} ({getattr(anchor_obj, 'label', '?')}) "
                f"[horiz={horizontal_distance:.2f}m, vert={delta[1]:.2f}m]"
            )
    return lines


def _run_structured_batch(
    gpt: Any,
    prompts: List[Any],
    images_list: Optional[List[Any]],
    *,
    response_model: Any,
    model: str,
    instructions: str,
    reasoning_effort: Optional[str] = None,
    detail: str = "auto",
) -> List[Any]:
    async def _runner() -> List[Any]:
        return await gpt.structured_prompt_batch(
            prompts,
            response_model=response_model,
            model=model,
            images=images_list,
            detail=detail,
            instructions=instructions,
            reasoning_effort=reasoning_effort,
        )

    try:
        return asyncio.run(_runner())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_runner())
        finally:
            loop.close()


def _walk_candidate_files(root: str, tokens: Sequence[str]) -> List[str]:
    matches: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            lower = filename.lower()
            if not lower.endswith((".json", ".jsonl", ".csv")):
                continue
            if tokens and not any(token in lower for token in tokens):
                continue
            matches.append(os.path.join(dirpath, filename))
    return sorted(matches)


def _preferred_scene_annotations_file(root: str, scene_name: str) -> Optional[str]:
    candidates = [
        os.path.join(root, "queries_by_scene_filtered", f"{scene_name}.json"),
        os.path.join(root, "queries_by_scene", f"{scene_name}.json"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def _entry_scene_id(entry: Dict[str, Any]) -> Optional[str]:
    for key in (
        "scene_id",
        "scan_id",
        "scene",
        "scan",
        "sceneId",
        "scanId",
        "scene_name",
        "scan_name",
    ):
        value = entry.get(key)
        if value:
            return str(value)
    return None


def _scene_matches(entry: Dict[str, Any], scene_name: str) -> bool:
    return _entry_scene_id(entry) == scene_name


def _iter_records(path: str) -> Iterable[Dict[str, Any]]:
    lower = path.lower()
    if lower.endswith(".csv"):
        with open(path, "r", encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)
        return

    with open(path, "r", encoding="utf-8") as handle:
        if lower.endswith(".jsonl"):
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
            return
        data = json.load(handle)

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
        return

    if isinstance(data, dict):
        for key in ("annotations", "records", "data", "items", "objects"):
            value = data.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
                return
        if any(isinstance(v, dict) for v in data.values()):
            for value in data.values():
                if isinstance(value, dict):
                    yield value


def _normalize_annotation(entry: Dict[str, Any], fallback_id: int) -> Optional[Dict[str, Any]]:
    utterance = (
        entry.get("utterance")
        or entry.get("description")
        or entry.get("query")
        or entry.get("text")
    )
    if not utterance:
        return None
    target_id = (
        entry.get("target_id")
        or entry.get("target")
        or entry.get("object_id")
        or entry.get("obj_id")
    )
    return {
        **entry,
        "ann_id": entry.get("ann_id") or entry.get("annotation_id") or entry.get("id") or fallback_id,
        "utterance": utterance,
        "target_id": None if target_id is None else str(target_id),
        "split": entry.get("split") or entry.get("eval_split") or "all",
        "scene_id": _entry_scene_id(entry),
    }


def _normalize_gt_object(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    object_id = entry.get("id") or entry.get("object_id") or entry.get("target_id")
    if object_id is None:
        return None

    center = entry.get("bbox_center") or entry.get("center")
    extent = entry.get("bbox_extent") or entry.get("extent") or entry.get("size")

    if center is None or extent is None:
        bbox = entry.get("bbox") or entry.get("box")
        if isinstance(bbox, dict):
            center = center or bbox.get("center")
            extent = extent or bbox.get("extent") or bbox.get("size")
        elif isinstance(bbox, (list, tuple)):
            arr = np.asarray(bbox, dtype=float)
            if arr.shape == (8, 3):
                lo = arr.min(axis=0)
                hi = arr.max(axis=0)
                center = ((lo + hi) / 2.0).tolist()
                extent = (hi - lo).tolist()
            elif arr.shape == (2, 3):
                lo, hi = arr
                center = ((lo + hi) / 2.0).tolist()
                extent = (hi - lo).tolist()

    if center is None or extent is None:
        return None

    return {
        **entry,
        "id": str(object_id),
        "bbox_center": [float(x) for x in center],
        "bbox_extent": [float(x) for x in extent],
        "label": entry.get("label") or entry.get("category") or entry.get("name") or "?",
    }


def _load_scene_annotations(scene_dir: str, nr3d_root: str) -> List[Dict[str, Any]]:
    if not os.path.isdir(nr3d_root):
        raise FileNotFoundError(f"NR3D root does not exist: {nr3d_root}")

    scene_name = _scene_base(scene_dir)
    preferred = _preferred_scene_annotations_file(nr3d_root, scene_name)
    files = [preferred] if preferred else _walk_candidate_files(
        nr3d_root, ("nr3d", "annotation", "annot")
    )
    annotations: List[Dict[str, Any]] = []

    for path in files:
        try:
            for idx, entry in enumerate(_iter_records(path)):
                if not _scene_matches(entry, scene_name):
                    continue
                normalized = _normalize_annotation(entry, len(annotations) + idx)
                if normalized is not None:
                    annotations.append(normalized)
        except Exception as exc:
            logger.debug("Skipping annotation file {}: {}", path, exc)

    if annotations:
        logger.info("Loaded {} annotations for {}", len(annotations), scene_name)
        return annotations

    raise FileNotFoundError(
        f"No NR3D annotations found for scene '{scene_name}' under {nr3d_root}"
    )


def _load_gt_scene_objects(scene_dir: str, nr3d_root: str) -> List[Dict[str, Any]]:
    if not os.path.isdir(nr3d_root):
        raise FileNotFoundError(f"NR3D root does not exist: {nr3d_root}")

    scene_name = _scene_base(scene_dir)
    files = _walk_candidate_files(
        nr3d_root, ("nr3d", "gt", "ground", "object", "bbox", scene_name.lower())
    )
    objects: List[Dict[str, Any]] = []

    for path in files:
        try:
            for entry in _iter_records(path):
                if not _scene_matches(entry, scene_name):
                    continue
                normalized = _normalize_gt_object(entry)
                if normalized is not None:
                    objects.append(normalized)
        except Exception as exc:
            logger.debug("Skipping GT file {}: {}", path, exc)

    if objects:
        deduped: Dict[str, Dict[str, Any]] = {}
        for obj in objects:
            deduped[obj["id"]] = obj
        result = list(deduped.values())
        logger.info("Loaded {} GT objects for {}", len(result), scene_name)
        return result

    raise FileNotFoundError(
        f"No GT scene objects found for scene '{scene_name}' under {nr3d_root}"
    )


def _annotation_split_map(annotations: Sequence[Dict[str, Any]]) -> Dict[Any, str]:
    mapping: Dict[Any, str] = {}
    for ann in annotations:
        mapping[ann.get("ann_id")] = ann.get("split") or "all"
    return mapping


def _annotation_flags(annotation: Optional[Dict[str, Any]], result: Dict[str, Any]) -> Dict[str, bool]:
    merged: Dict[str, bool] = {}
    for field in _ANNOTATION_FLAG_FIELDS:
        if annotation is not None and field in annotation:
            merged[field] = bool(annotation.get(field))
        else:
            merged[field] = bool(result.get(field))
    return merged


def _compute_eval_stats(
    rows: Sequence[Dict[str, Any]],
    iou_thresholds: Sequence[float],
) -> Dict[str, Any]:
    ious = [float(row.get("iou_3d_strict", 0.0)) for row in rows]
    num_predictions = sum(row.get("predicted_object_id") is not None for row in rows)

    stats: Dict[str, Any] = {
        "num_queries": len(rows),
        "num_predictions": num_predictions,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
    }
    for threshold in iou_thresholds:
        key = f"acc@{threshold:g}"
        stats[key] = (
            float(sum(iou >= threshold for iou in ious) / len(ious))
            if ious
            else 0.0
        )
    return stats


def _compute_grounding_metrics(
    results: Sequence[Dict[str, Any]],
    annotations: Sequence[Dict[str, Any]],
    gt_corners_map: Dict[str, np.ndarray],
    iou_thresholds: Sequence[float],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    split_map = _annotation_split_map(annotations)
    ann_by_id = {ann.get("ann_id"): ann for ann in annotations}

    enriched_results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    split_grouped: Dict[str, List[Dict[str, Any]]] = {"overall": []}
    bucket_grouped: Dict[str, List[Dict[str, Any]]] = {
        bucket_name: [] for bucket_name, _, _ in _LANGUAGE_BUCKETS
    }

    for result in results:
        annotation = ann_by_id.get(result.get("ann_id"))
        flags = _annotation_flags(annotation, result)
        gt_bbox = gt_corners_map.get(str(result.get("ground_truth_target_id")))
        pred_bbox = result.get("bbox_3d")
        iou = (
            strict_box3d_iou(pred_bbox, gt_bbox)
            if gt_bbox is not None and pred_bbox is not None
            else 0.0
        )

        enriched = {**result, **flags, "iou_3d_strict": iou}
        enriched_results.append(enriched)
        split_grouped["overall"].append(enriched)

        split = split_map.get(result.get("ann_id"), "all")
        split_grouped.setdefault(split, []).append(enriched)

        for bucket_name, field_name, expected in _LANGUAGE_BUCKETS:
            if flags[field_name] is expected:
                bucket_grouped[bucket_name].append(enriched)

        if iou < 0.1:
            failures.append(enriched)

    ordered_groups: Dict[str, List[Dict[str, Any]]] = {}
    ordered_groups.update(split_grouped)
    ordered_groups.update(bucket_grouped)

    metrics = {
        "iou_metric": "strict_axis_aligned_iou",
        "iou_thresholds": list(iou_thresholds),
        "splits": {
            group_name: _compute_eval_stats(rows, iou_thresholds)
            for group_name, rows in ordered_groups.items()
        },
    }
    return enriched_results, metrics, failures


def _evaluate_results(
    results: Sequence[Dict[str, Any]],
    annotations: Sequence[Dict[str, Any]],
    gt_corners_map: Dict[str, np.ndarray],
    iou_thresholds: Sequence[float],
) -> Dict[str, Any]:
    split_map = _annotation_split_map(annotations)
    grouped: Dict[str, List[Dict[str, Any]]] = {"overall": []}

    for result in results:
        ann_id = result.get("ann_id")
        split = split_map.get(ann_id, "all")
        grouped.setdefault(split, []).append(result)
        grouped["overall"].append(result)

    def _compute(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        ious: List[float] = []
        predictions = 0
        for item in items:
            if item.get("predicted_object_id") is not None:
                predictions += 1
            gt_bbox = gt_corners_map.get(str(item.get("ground_truth_target_id")))
            pred_bbox = _bbox_to_np(item.get("bbox_3d"))
            ious.append(strict_box3d_iou(pred_bbox, gt_bbox) if gt_bbox is not None and pred_bbox is not None else 0.0)

        stats: Dict[str, Any] = {
            "num_queries": len(items),
            "num_predictions": predictions,
            "mean_iou": float(np.mean(ious)) if ious else 0.0,
        }
        for threshold in iou_thresholds:
            key = f"acc@{threshold:g}"
            stats[key] = float(sum(iou >= threshold for iou in ious) / len(ious)) if ious else 0.0
        return stats

    metrics = {"iou_thresholds": list(iou_thresholds), "splits": {}}
    for split, items in grouped.items():
        metrics["splits"][split] = _compute(items)
    return metrics


def _format_metrics(metrics: Dict[str, Any]) -> str:
    lines: List[str] = []
    for split, stats in metrics.get("splits", {}).items():
        parts = [f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}" for key, value in stats.items()]
        lines.append(f"{split}: " + ", ".join(parts))
    return "\n".join(lines)


def _write_debug_entry(
    debug_file: TextIO,
    query_idx: int,
    ann: Dict[str, Any],
    utterance: str,
    context_text: str,
    selection: Any,
    pred_id: Optional[str],
    bbox: Optional[np.ndarray],
    gt_corners_map: Optional[Dict[str, np.ndarray]],
    gt_label_map: Optional[Dict[str, str]],
    *,
    frame_results: Optional[Dict[str, Sequence[Any]]] = None,
    images: Optional[Sequence[Any]] = None,
) -> None:
    gt_id = str(ann.get("target_id"))
    gt_label = (gt_label_map or {}).get(gt_id, "?")
    gt_bbox = (gt_corners_map or {}).get(gt_id)
    iou = strict_box3d_iou(bbox, gt_bbox) if bbox is not None and gt_bbox is not None else 0.0

    debug_file.write(f"## Query {query_idx}\n")
    debug_file.write(f"ann_id: {ann.get('ann_id')}\n")
    debug_file.write(f"utterance: {utterance}\n")
    debug_file.write(f"gt_target_id: {gt_id} ({gt_label})\n")
    debug_file.write(f"predicted_id: {pred_id}\n")
    debug_file.write(f"confidence: {getattr(selection, 'confidence', None)}\n")
    debug_file.write(f"reason: {getattr(selection, 'reason', '')}\n")
    debug_file.write(f"iou: {iou:.4f}\n")
    debug_file.write(f"images_used: {len(images or [])}\n")
    if frame_results:
        for modality, hits in frame_results.items():
            ids = [hit.chunk.id for hit in hits[:5]]
            debug_file.write(f"{modality}: {ids}\n")
    debug_file.write("context:\n")
    debug_file.write(context_text + "\n\n")


def _collect_failed_queries(
    results: Sequence[Dict[str, Any]],
    annotations: Sequence[Dict[str, Any]],
    gt_corners_map: Dict[str, np.ndarray],
    iou_threshold: float = 0.1,
) -> List[Dict[str, Any]]:
    ann_by_id = {ann.get("ann_id"): ann for ann in annotations}
    failed: List[Dict[str, Any]] = []
    for result in results:
        gt_bbox = gt_corners_map.get(str(result.get("ground_truth_target_id")))
        pred_bbox = _bbox_to_np(result.get("bbox_3d"))
        iou = strict_box3d_iou(pred_bbox, gt_bbox) if gt_bbox is not None and pred_bbox is not None else 0.0
        if iou < iou_threshold:
            failed.append(
                {
                    "result": result,
                    "annotation": ann_by_id.get(result.get("ann_id")),
                    "iou": iou,
                }
            )
    return failed


def _save_experiment_artifacts(
    output_dir: str,
    scene_dir: str,
    run_name: str,
    args: argparse.Namespace,
    script_path: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    paths = _eval_output_paths(output_dir, scene_dir, run_name, script_path=script_path)
    args_path = paths["args"]
    meta_path = paths["meta"]
    with open(args_path, "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "python": sys.executable,
                "cwd": os.getcwd(),
                "script_path": os.path.abspath(script_path),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    try:
        shutil.copy2(script_path, paths["script_copy"])
    except Exception as exc:
        logger.debug("Could not copy experiment script {}: {}", script_path, exc)


def _write_outputs(
    output_dir: str,
    scene_dir: str,
    run_name: str,
    results: Sequence[Dict[str, Any]],
    metrics: Dict[str, Any],
    annotations: Sequence[Dict[str, Any]],
    gt_corners_map: Dict[str, np.ndarray],
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    base = _scene_base(scene_dir)
    results_path = os.path.join(output_dir, f"{base}_{run_name}_results.json")
    metrics_path = os.path.join(output_dir, f"{base}_{run_name}_metrics.json")
    failed_path = os.path.join(output_dir, f"{base}_{run_name}_failed.json")
    summary_path = os.path.join(output_dir, f"{base}_{run_name}_summary.txt")

    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(list(results), handle, indent=2)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with open(failed_path, "w", encoding="utf-8") as handle:
        json.dump(
            _collect_failed_queries(results, annotations, gt_corners_map),
            handle,
            indent=2,
        )
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(_format_metrics(metrics) + "\n")

    logger.info("Wrote evaluation results to {}", output_dir)
