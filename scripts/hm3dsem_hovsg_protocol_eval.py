"""Score KeySG graphs on HM3DSem hierarchical object retrieval with the HOV-SG protocol.

Mirrors HOV-SG's application/eval/eval_long_queries.py step for step, swapping only the
graph and the encoder:

    parse   regex on the templated query "<obj> in the <room> on floor <n>" (same as HOV-SG)
    floor   n-th KeySG floor (floors are numbered bottom-up); all rooms if that floor is empty
    room    SigLIP2 text(room) vs each room's keyframe SigLIP2 embeddings, max over views, top-3
            (HOV-SG: CLIP text vs room view embeddings, max over views, top-3)
    object  SigLIP2 text([obj, "background"]) vs object features of the selected rooms; keep
            objects whose argmax is obj, sorted by score, falling back to plain top-k when none
            pass (HOV-SG Graph.query_object with negative_prompt=["background"])
    IoU     axis-aligned box IoU of predicted vs GT point clouds (hovsg compute_3d_iou)

The encoder is the one KeySG was built with (hf-hub:timm/ViT-gopt-16-SigLIP2-384): object
features and keyframe embeddings are read from the saved graph, only text is encoded here.

Example:
    python scripts/hm3dsem_hovsg_protocol_eval.py \
        --queries /mnt/Backup2nd/Dataset/hm3dsem/long_queries_hovsg20_obj_room_floor.json \
        --keysg-root output/hm3dsem/HMP3D --scenes 00824-Dd4bFSTQ8gi 00829-QaLdnwvtxbs \
        --out output/hm3dsem_eval/hovsg20_orf.json

    python scripts/hm3dsem_hovsg_protocol_eval.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np

TOP_KS = (1, 5, 10)
TAUS = (0.0, 0.10, 0.50)
QUERY_RE = re.compile(r"^(?P<obj>.+?) in the (?P<room>.+?)(?: on floor (?P<floor>\d+))?$")


def parse_query(query):
    """"couch in the living room on floor 0" -> (0, "living room", "couch")."""
    m = QUERY_RE.match(query)
    if m is None:
        raise ValueError(f"unparsable query: {query!r}")
    floor = m.group("floor")
    return (int(floor) if floor is not None else None, m.group("room"), m.group("obj"))


def _unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def rank_rooms(room_q, room_views, top_k):
    """room_views: {room_id: (V, D) unit views}. Max over views, best first."""
    scored = [(rid, float(np.max(views @ room_q))) for rid, views in room_views.items() if len(views)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [rid for rid, _ in scored[:top_k]]


def rank_objects(text_feats, obj_feats, top_k):
    """text_feats: (1 + n_negative, D), row 0 is the query. Returns indices into obj_feats."""
    sim = text_feats @ obj_feats.T  # (C, N)
    order = np.argsort(-sim[0])[:top_k]
    keep = np.where(np.argmax(sim, axis=0) == 0)[0]
    if len(keep):
        order = keep[np.argsort(-sim[:, keep].max(axis=0))][:top_k]
    return order.tolist()


def aabb_iou(pts1, pts2):
    lo1, hi1, lo2, hi2 = pts1.min(0), pts1.max(0), pts2.min(0), pts2.max(0)
    inter = np.prod(np.maximum(np.minimum(hi1, hi2) - np.maximum(lo1, lo2), 0.0))
    return float(inter / (np.prod(hi1 - lo1) + np.prod(hi2 - lo2) - inter))


def hits_for(ious):
    """{(K, tau): hit} from the per-rank best IoUs. tau=0 means "any overlap"."""
    return {
        (k, tau): any((iou > 0.0 if tau == 0.0 else iou >= tau) for iou in ious[:k])
        for k in TOP_KS
        for tau in TAUS
    }


def aggregate(per_query):
    if not per_query:
        return {}
    return {
        f"R@{k}_IoU>={tau}": 100.0 * sum(q[(k, tau)] for q in per_query) / len(per_query)
        for k in TOP_KS
        for tau in TAUS
    }


class SceneIndex:
    """Floors, per-room keyframe views and object features of one KeySG output dir."""

    def __init__(self, scene_dir):
        from keysg.graph import KeySGGraph

        g = KeySGGraph.from_output_dir(scene_dir, build_rag=False)
        self.floor_rooms = [[r.id for r in f.rooms] for f in sorted(g.floors, key=lambda f: int(f.id))]
        self.room_objects = {
            r.id: [o for o in r.objects if o.feature is not None and o.pcd is not None]
            for r in g.rooms.values()
        }

        cache = os.path.join(scene_dir, "rag_cache")
        chunks = json.load(open(os.path.join(cache, "graph_chunks_meta.json")))["chunks"]
        meta = json.load(open(os.path.join(cache, "graph_frame_visual_meta.json")))
        embs = _unit(np.load(os.path.join(cache, "graph_frame_visual_embeddings.npy")))
        views = defaultdict(list)
        for emb, ci in zip(embs, meta["frame_chunk_indices"]):
            views[chunks[ci]["metadata"]["room_id"]].append(emb)
        self.room_views = {rid: np.stack(v) for rid, v in views.items()}
        self.clip_model_id = meta["clip_model_id"]

    def rooms_on_floor(self, floor):
        if floor is None or floor >= len(self.floor_rooms) or not self.floor_rooms[floor]:
            return [rid for rooms in self.floor_rooms for rid in rooms]
        return self.floor_rooms[floor]

    def query(self, encode, query, top_k, room_topk):
        floor, room_q, obj_q = parse_query(query)
        on_floor = self.rooms_on_floor(floor)
        room_ids = rank_rooms(
            encode([room_q])[0], {rid: self.room_views.get(rid, []) for rid in on_floor}, room_topk
        )
        objects = [o for rid in room_ids for o in self.room_objects.get(rid, [])]
        if not objects:
            return [], room_ids
        order = rank_objects(encode([obj_q, "background"]), _unit([o.feature for o in objects]), top_k)
        return [objects[i] for i in order], room_ids


def run(args):
    import open3d as o3d
    from keysg.utils.clip_utils import CLIPFeatureExtractor

    data = json.load(open(args.queries))
    by_scene = defaultdict(list)
    for entry in data["queries"]:
        by_scene[entry["scene"]].append(entry)

    clip, text_cache = None, {}

    def encode(texts):
        missing = [t for t in texts if t not in text_cache]
        if missing:
            for t, f in zip(missing, clip.get_text_feats(missing)):
                text_cache[t] = f
        return _unit([text_cache[t] for t in texts])

    per_query, per_scene_hits, rows, gt_cache = [], defaultdict(list), [], {}
    for scene in args.scenes or sorted(by_scene):
        scene_dir = os.path.join(args.keysg_root, scene)
        if not os.path.isdir(scene_dir):
            print(f"[skip] {scene}: no KeySG output at {scene_dir}")
            continue
        index = SceneIndex(scene_dir)
        if clip is None:
            clip = CLIPFeatureExtractor({"model_name": index.clip_model_id, "device": args.device})
        entries = by_scene[scene]
        print(f"[{scene}] {len(entries)} queries, floors={[len(f) for f in index.floor_rooms]} rooms")

        for entry in entries:
            preds, room_ids = index.query(encode, entry["query"], args.top_k, args.room_topk)
            gt_pts = []
            for g in entry["gt_objects"]:
                if g["pcd"] not in gt_cache:
                    pcd = o3d.io.read_point_cloud(os.path.join(data["data_root"], g["pcd"]))
                    gt_cache[g["pcd"]] = np.asarray(pcd.points)
                gt_pts.append(gt_cache[g["pcd"]])
            ious = [max(aabb_iou(np.asarray(p.pcd.points), gt) for gt in gt_pts) for p in preds]
            hits = hits_for(ious)
            per_query.append(hits)
            per_scene_hits[scene].append(hits)
            rows.append({
                "scene": scene,
                "query": entry["query"],
                "rooms": room_ids,
                "pred_object_ids": [p.id for p in preds],
                "pred_labels": [p.label for p in preds],
                "ious": ious,
                "hits": {f"{k}|{tau}": v for (k, tau), v in hits.items()},
            })

    metrics = aggregate(per_query)
    print(f"\n=== {len(per_query)} queries over {len(per_scene_hits)} scenes ===")
    for name, value in metrics.items():
        print(f"{name:24s} {value:6.2f}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "config": vars(args),
                "num_queries": len(per_query),
                "metrics": metrics,
                "per_scene": {s: aggregate(h) for s, h in per_scene_hits.items()},
                "results": rows,
            }, f, indent=1)
        print(f"wrote {args.out}")


def self_test():
    assert parse_query("rug in the living room on floor 1") == (1, "living room", "rug")
    assert parse_query("tv in the family room") == (None, "family room", "tv")

    e = np.eye(3, dtype=np.float32)
    views = {"a": e[[1]], "b": e[[0, 2]], "c": np.zeros((0, 3))}
    assert rank_rooms(e[0], views, 3) == ["b", "a"], "max over views, empty rooms skipped"

    # objects 0 and 2 look like the query, object 1 looks like background
    text = e[:2]  # row 0 query, row 1 "background"
    objs = _unit([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.6, 0.4, 0.0]])
    assert rank_objects(text, objs, 10) == [0, 2], "background-labelled object is dropped"
    assert rank_objects(text, objs[[1]], 10) == [0], "falls back to plain ranking when none pass"

    box = np.array([[0, 0, 0], [1, 1, 1]], float)
    assert abs(aabb_iou(box, box + [0.5, 0, 0]) - 1 / 3) < 1e-9
    hits = hits_for([0.0, 1 / 3])
    assert not hits[(1, 0.0)] and hits[(5, 0.1)] and not hits[(5, 0.5)]
    print("self-test ok")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", help="long_queries_*.json from HOV-SG's dump_long_queries.py")
    parser.add_argument("--keysg-root", help="dir with one KeySG output dir per scene")
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--top-k", type=int, default=max(TOP_KS))
    parser.add_argument("--room-topk", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.queries or not args.keysg_root:
        parser.error("--queries and --keysg-root are required (or use --self-test)")
    run(args)


if __name__ == "__main__":
    main()
