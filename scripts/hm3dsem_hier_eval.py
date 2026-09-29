"""Hierarchical object retrieval on HM3DSem using the KeySG w/ RAG cascade.

The official HM3DSem evaluation (``sim_search.py`` and its w/ RAG counterpart) is not part
of the public KeySG release, so this script reconstructs the cascade described in the paper
(Sec. III-E):

    P(C|q) ~ P(F|q) P(R|F,q) * prod_o P(o|R,q) * prod_s P(s|R,q)

Cosine similarities approximate the conditionals, top-k is kept at each level, and objects
are filtered by the rooms identified one level up.

Everything the paper leaves unspecified is exposed as a flag (defaults in parentheses):

    --prior     hard | soft        location prior as a filter, or as a score multiplier (hard)
    --modality  object_visual | text | mean    what ranks the objects        (object_visual)
    --k-floor / --k-room           candidates kept per level                 (1 / 100)
    --pool                         candidates fetched per doc_type           (200)

The keyframe term prod_s P(s|R,q) is NOT implemented: the released code never turns frame
hits into object scores (``node_tags`` only ever reaches chunk text, metadata and the UI),
so any weighting would be invented rather than reconstructed.

Ground truth comes from HOV-SG's ``application/eval/dump_long_queries.py`` and is scored
with axis-aligned box IoU, the same metric as ``scripts/nr3d_eval.py``.

Example:
    python scripts/hm3dsem_hier_eval.py \
        --queries /mnt/Backup2nd/Dataset/hm3dsem/long_queries_obj_room_floor.json \
        --keysg-root output/keysg_rag1/HMP3D \
        --out output/hm3dsem_hier_eval.json

    python scripts/hm3dsem_hier_eval.py --self-test   # scoring logic only, no scene needed
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

TOP_KS = (1, 5, 10)
TAUS = (0.0, 0.10, 0.50)


# --------------------------------------------------------------------------------------
# retrieval cascade
# --------------------------------------------------------------------------------------
def _level_hits(retriever, query: str, doc_type: str, pool: int, visual: bool) -> List[Tuple[dict, float]]:
    """Return [(chunk_metadata, score), ...] for one doc_type, best score first.

    ``search`` filters by doc_type only AFTER the FAISS lookup, so `pool` has to be large
    enough that chunks of this type survive the cut (object chunks dominate a scene).
    """
    results = retriever.search(
        query,
        top_k=pool,
        doc_types=[doc_type],
        frame_modality="text",  # never load the CLIP encoder for frame chunks
        object_modality="visual" if visual else "text",
    )
    key = "object_visual" if visual else "text"
    return [(r.chunk.metadata, float(r.score)) for r in results.get(key, [])]


def _object_scores(retriever, query: str, pool: int, modality: str) -> List[Tuple[dict, float]]:
    if modality in ("text", "object_visual"):
        return _level_hits(retriever, query, "object", pool, visual=modality == "object_visual")

    # "mean": average the two modalities over the union of retrieved objects
    merged: Dict[str, List[float]] = defaultdict(list)
    meta_by_id: Dict[str, dict] = {}
    for visual in (False, True):
        for meta, score in _level_hits(retriever, query, "object", pool, visual=visual):
            oid = meta.get("object_id")
            if oid is None:
                continue
            meta_by_id[oid] = meta
            merged[oid].append(score)
    scored = [(meta_by_id[oid], float(np.mean(scores))) for oid, scores in merged.items()]
    return sorted(scored, key=lambda x: x[1], reverse=True)


def retrieve(retriever, query: str, cfg: argparse.Namespace) -> List[Tuple[str, float]]:
    """Run floor -> room -> object and return ranked (object_id, score)."""
    floors = _level_hits(retriever, query, "floor", cfg.pool, visual=False)[: cfg.k_floor]
    floor_score = {str(m.get("floor_id")): s for m, s in floors}

    rooms = _level_hits(retriever, query, "room", cfg.pool, visual=False)
    if cfg.prior == "hard" and floor_score:
        rooms = [(m, s) for m, s in rooms if str(m.get("floor_id")) in floor_score]
    rooms = rooms[: cfg.k_room]
    room_score = {str(m.get("room_id")): s for m, s in rooms}

    objects = _object_scores(retriever, query, cfg.pool, cfg.modality)
    ranked: List[Tuple[str, float]] = []
    for meta, score in objects:
        room_id = str(meta.get("room_id"))
        oid = meta.get("object_id")
        if oid is None:
            continue
        if cfg.prior == "hard":
            if room_id not in room_score:
                continue
        else:  # soft: the location prior re-weights instead of pruning
            room_meta = next((m for m, _ in rooms if str(m.get("room_id")) == room_id), None)
            floor_id = str(room_meta.get("floor_id")) if room_meta else None
            score *= room_score.get(room_id, 0.0) * floor_score.get(floor_id, 1.0)
        ranked.append((oid, score))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[: max(TOP_KS)]


# --------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------
def _hit(iou: float, tau: float) -> bool:
    # tau == 0 is reported by the paper as a threshold, but "IoU >= 0" holds for every box;
    # it is read here as "any overlap at all".
    return iou > 0.0 if tau == 0.0 else iou >= tau


def score_query(
    pred_boxes: Sequence[Any],
    gt_boxes: Sequence[Any],
    iou_fn,
) -> Dict[Tuple[int, float], bool]:
    """Return {(K, tau): hit} for one query, given predictions ranked best first."""
    ious = [max((iou_fn(p, g) for g in gt_boxes), default=0.0) for p in pred_boxes]
    return {
        (k, tau): any(_hit(iou, tau) for iou in ious[:k])
        for k in TOP_KS
        for tau in TAUS
    }


def aggregate(per_query: Sequence[Dict[Tuple[int, float], bool]]) -> Dict[str, float]:
    if not per_query:
        return {}
    return {
        f"R@{k}_IoU>={tau}": 100.0 * sum(q[(k, tau)] for q in per_query) / len(per_query)
        for k in TOP_KS
        for tau in TAUS
    }


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------
def run(cfg: argparse.Namespace) -> Dict[str, Any]:
    from keysg.graph import KeySGGraph
    from keysg.utils.iou_eval import bbox_to_corners, strict_box3d_iou

    data = json.loads(open(cfg.queries).read())
    by_scene: Dict[str, List[dict]] = defaultdict(list)
    for entry in data["queries"]:
        by_scene[entry["scene"]].append(entry)

    per_query, rows = [], []
    for scene, entries in sorted(by_scene.items()):
        scene_dir = os.path.join(cfg.keysg_root, scene)
        if not os.path.isdir(scene_dir):
            print(f"[skip] no KeySG output for {scene} at {scene_dir}")
            continue
        print(f"[{scene}] {len(entries)} queries")
        graph = KeySGGraph.from_output_dir(scene_dir, build_rag=True)
        retriever = graph._ensure_rag()

        for entry in entries:
            ranked = retrieve(retriever, entry["query"], cfg)
            pred_boxes = [
                graph.objects[oid].bbox_3d
                for oid, _ in ranked
                if oid in graph.objects and graph.objects[oid].bbox_3d is not None
            ]
            gt_boxes = [
                bbox_to_corners(np.concatenate([g["aabb_center"], g["aabb_dims"]]))
                for g in entry["gt_objects"]
            ]
            hits = score_query(pred_boxes, gt_boxes, strict_box3d_iou)
            per_query.append(hits)
            rows.append({
                "scene": scene,
                "query": entry["query"],
                "pred_object_ids": [oid for oid, _ in ranked],
                "hits": {f"{k}|{tau}": v for (k, tau), v in hits.items()},
            })

    metrics = aggregate(per_query)
    out = {
        "config": vars(cfg),
        "num_queries": len(per_query),
        "metrics": metrics,
        "results": rows,
    }
    if cfg.out:
        with open(cfg.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {cfg.out}")
    for name, value in metrics.items():
        print(f"{name:20s} {value:6.2f}")
    return out


def self_test() -> None:
    """Scoring logic only: no scene graph, no API key, no model downloads."""
    box = lambda cx: [cx - 0.5, 0.0, 0.0, cx + 0.5, 1.0, 1.0]  # noqa: E731  (lo/hi pairs)

    def iou(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        lo = np.maximum(a[:3], b[:3])
        hi = np.minimum(a[3:], b[3:])
        inter = float(np.prod(np.maximum(hi - lo, 0.0)))
        if inter <= 0:
            return 0.0
        va = float(np.prod(a[3:] - a[:3]))
        vb = float(np.prod(b[3:] - b[:3]))
        return inter / (va + vb - inter)

    gt = [box(0.0)]
    # rank 1 misses entirely, rank 2 overlaps by 0.5/1.5 = 1/3, rank 3 is exact
    preds = [box(10.0), box(0.5), box(0.0)]
    hits = score_query(preds, gt, iou)
    assert hits[(1, 0.0)] is False, "top-1 is a disjoint box"
    assert hits[(5, 0.0)] is True, "top-5 contains an overlapping box"
    assert hits[(5, 0.10)] is True, "IoU 1/3 clears tau=0.1"
    assert hits[(5, 0.50)] is True, "the exact box clears tau=0.5"

    only_partial = score_query(preds[:2], gt, iou)
    assert only_partial[(5, 0.50)] is False, "IoU 1/3 must not clear tau=0.5"

    agg = aggregate([hits, only_partial])
    assert agg["R@5_IoU>=0.5"] == 50.0, agg
    assert agg["R@1_IoU>=0.0"] == 0.0, agg
    print("self-test ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", help="long_queries_*.json from HOV-SG's dump_long_queries.py")
    parser.add_argument("--keysg-root", help="directory holding one built KeySG output per scene")
    parser.add_argument("--out", default=None)
    parser.add_argument("--prior", choices=["hard", "soft"], default="hard")
    parser.add_argument("--modality", choices=["object_visual", "text", "mean"], default="object_visual")
    parser.add_argument("--k-floor", type=int, default=1)
    parser.add_argument("--k-room", type=int, default=100)
    parser.add_argument("--pool", type=int, default=200)
    parser.add_argument("--self-test", action="store_true")
    cfg = parser.parse_args()

    if cfg.self_test:
        self_test()
        return
    if not cfg.queries or not cfg.keysg_root:
        parser.error("--queries and --keysg-root are required (or use --self-test)")
    run(cfg)


if __name__ == "__main__":
    main()
