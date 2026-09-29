"""Generate IRef-VLA-style, room-conditioned referential queries for HM3DSem walks.

Input : <data_root>/<scene>/scene_info.json (HOV-SG GT; Y-up, AABBs, region labels)
Output: <out_dir>/<scene>_all.json         every valid (target, relation, anchors) combo
        <out_dir>/<scene>_sampled<N>.json  diversity-first subset

Every query has exactly one answer inside its room:
  * target/anchor classes exclude structure and vague labels
  * anchors are the only object of their class in the room
  * no same-class distractor satisfies a *relaxed* version of the relation
  * ordinal relations need a distance margin to both neighbours
Rooms labelled empty/unknown are skipped.
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

SCENES = ["00824-Dd4bFSTQ8gi", "00829-QaLdnwvtxbs", "00843-DYehNKdT76V", "00861-GLAQ4DNUx5U",
          "00862-LT9Jq6dN3Ea", "00873-bxsVRursffK", "00877-4ok3usBNeis", "00890-6s7QHgap2fW"]
SKIP_ROOMS = {"", "unknown", "unknown room", "empty room"}
EXCLUDE = {
    # structure
    "wall", "ceiling", "floor", "door frame", "window frame", "shower wall", "beam", "support beam",
    "ceiling light fixture connection", "wall panel", "parapet", "pipe", "stairs", "handrail",
    "stairs railing", "railing", "baseboard", "column", "pillar", "window glass", "door hinge",
    "ceiling molding", "skirting board", "floor mat frame",
    # vague
    "unknown", "clutter", "object", "misc", "stuff", "appliance", "device", "decoration",
    "bathroom accessory", "bathroom utensil", "washing stuff", "basket of something", "wall electronics",
}
MIN_SIZE = 0.05  # m, largest AABB side
PHRASES = {
    "near": ["near", "next to", "close to", "beside", "adjacent to"],
    "above": ["above", "over"],
    "below": ["below", "under", "beneath", "underneath"],
    "on": ["on", "on top of"],
    "in": ["in", "inside", "within"],
    "closest": ["closest to", "nearest to"],
    "farthest": ["farthest from", "most distant from"],
    "between": ["between", "in between", "in the middle of"],
}
UP = 1  # Y-up
GROUPS = {"near": "proximity", "above": "vertical", "below": "vertical", "on": "support", "in": "containment",
          "between": "between", "closest": "superlative", "farthest": "superlative"}


class Box:
    def __init__(self, o):
        c, d = np.asarray(o["aabb_center"], float), np.asarray(o["aabb_dims"], float)
        self.id, self.cls, self.c, self.lo, self.hi = o["id"], o["category"], c, c - d / 2, c + d / 2
        self.vol = float(np.prod(d))
        self.foot = float(d[0] * d[2])


def gap(a, b):
    return float(np.linalg.norm(np.maximum(0, np.maximum(a.lo - b.hi, b.lo - a.hi))))


def foot_overlap(a, b):
    """Horizontal (x,z) overlap area."""
    w = [max(0.0, min(a.hi[i], b.hi[i]) - max(a.lo[i], b.lo[i])) for i in (0, 2)]
    return w[0] * w[1]


def inside_frac(t, a):
    w = np.maximum(0, np.minimum(t.hi, a.hi) - np.maximum(t.lo, a.lo))
    return float(np.prod(w)) / max(t.vol, 1e-9)


def dist(a, b):
    return float(np.linalg.norm(a.c - b.c))


# Binary relations: strict test for the target, relaxed test that every distractor must fail.
def near(t, a, relaxed=False):
    return gap(t, a) <= (1.0 if relaxed else 0.5)


def above(t, a, relaxed=False):
    v = t.lo[UP] - a.hi[UP]
    ov = foot_overlap(t, a) / max(min(t.foot, a.foot), 1e-9)
    return (-0.1 <= v <= 2.0 and ov >= 0.1) if relaxed else (0.1 <= v <= 1.5 and ov >= 0.3)


def below(t, a, relaxed=False):
    return above(a, t, relaxed)


def on(t, a, relaxed=False):
    v = abs(t.lo[UP] - a.hi[UP])
    cover = foot_overlap(t, a) / max(t.foot, 1e-9)
    if relaxed:
        return v <= 0.2 and cover >= 0.3
    return v <= 0.1 and cover >= 0.7 and t.foot < a.foot


def inside(t, a, relaxed=False):
    return inside_frac(t, a) >= (0.5 if relaxed else 0.9) and (relaxed or a.vol >= 3 * t.vol)


BINARY = {"near": near, "above": above, "below": below, "on": on, "in": inside}
ORDINAL = {"closest": (0, False), "farthest": (0, True)}  # superlatives; 2nd/3rd dropped as unnatural
MARGIN = 0.3  # m, ordinal distance gap to neighbours


def between(t, a1, a2, relaxed=False):
    p, q, x = a1.c[[0, 2]], a2.c[[0, 2]], t.c[[0, 2]]
    seg = q - p
    L = float(np.linalg.norm(seg))
    if not 0.5 <= L <= 5.0:
        return False
    s = float(np.dot(x - p, seg) / L**2)
    perp = float(abs(seg[0] * (x - p)[1] - seg[1] * (x - p)[0]) / L)
    if relaxed:
        return 0.1 <= s <= 0.9 and perp <= 0.4 * L
    return 0.2 <= s <= 0.8 and perp <= min(0.25 * L, 1.0)


def statement(t_cls, phrase, anchors):
    verb = "are" if t_cls.endswith("s") and not t_cls.endswith("ss") else "is"
    obj = " and ".join(f"the {a.cls}" for a in anchors)
    return f"the {t_cls} that {verb} {phrase} {obj}"


def region_queries(objs, rng):
    by_cls = defaultdict(list)
    for o in objs:
        by_cls[o.cls].append(o)
    anchors = [v[0] for v in by_cls.values() if len(v) == 1]
    out = []

    def add(t, rel, ancs):
        out.append((t, rel, ancs, [d for d in by_cls[t.cls] if d is not t]))

    for a in anchors:
        for cls, group in by_cls.items():
            if cls == a.cls:
                continue
            for rel, fn in BINARY.items():
                for t in group:
                    if fn(t, a) and not any(fn(d, a, relaxed=True) for d in group if d is not t):
                        add(t, rel, [a])
            ds = sorted((dist(t, a), i) for i, t in enumerate(group))
            for rel, (k, far) in ORDINAL.items():
                if len(group) < k + 2:  # k-th needs at least one object beyond it
                    continue
                order = ds[::-1] if far else ds
                gaps = [abs(order[k][0] - order[j][0]) for j in (k - 1, k + 1) if 0 <= j < len(order)]
                if min(gaps) >= MARGIN:
                    add(group[order[k][1]], rel, [a])
    for a1, a2 in combinations(anchors, 2):
        for cls, group in by_cls.items():
            if cls in (a1.cls, a2.cls):
                continue
            for t in group:
                if between(t, a1, a2) and not any(between(d, a1, a2, True) for d in group if d is not t):
                    add(t, "between", [a1, a2])
    return out


def box_json(b):
    return {"id": b.id, "class": b.cls, "center": b.c.tolist(), "dims": (b.hi - b.lo).tolist()}


def generate(scene_info, rng):
    rooms = {r["id"]: r for r in scene_info["regions"]}
    objs = defaultdict(list)
    for o in scene_info["objects"]:
        if o["category"] not in EXCLUDE and max(o["aabb_dims"]) >= MIN_SIZE:
            objs[o["region_id"]].append(Box(o))
    queries = []
    for rid in sorted(objs):
        room = rooms.get(rid)
        if room is None or room["category"].strip().lower() in SKIP_ROOMS:
            continue
        for t, rel, ancs, dis in region_queries(objs[rid], rng):
            stmt = statement(t.cls, rng.choice(PHRASES[rel]), ancs)
            queries.append({
                "query": f"In the {room['category']}, {stmt}",
                "statement": stmt,
                "relation": rel,
                "relation_type": "ternary" if rel == "between" else "binary",
                "region_id": rid,
                "region_label": room["category"],
                "floor_id": room.get("floor_id"),
                "target_id": t.id,
                "target_class": t.cls,
                "target": box_json(t),
                "anchors": [box_json(a) for a in ancs],
                "anchor_classes": [a.cls for a in ancs],
                "distractor_ids": [d.id for d in dis],
            })
    return queries


def sample(queries, n, rng):
    """Rotate relation groups (least-picked first), then the group's least-picked relation; within
    it pick the candidate adding the most unseen target class / instance / anchor class / room,
    preferring ones with distractors."""
    by_rel = defaultdict(list)
    for q in queries:
        by_rel[q["relation"]].append(q)
    for v in by_rel.values():
        rng.shuffle(v)

    def feats(q):
        return ([("t", q["target_class"]), ("i", q["target_id"]), ("r", q["region_id"])]
                + [("a", a) for a in q["anchor_classes"]])

    seen, rel_n, grp_n, out = Counter(), Counter(), Counter(), []
    while len(out) < n and any(by_rel.values()):
        live = [r for r in by_rel if by_rel[r]]
        grp = min({GROUPS[r] for r in live}, key=lambda g: (grp_n[g], g))
        rel = min((r for r in live if GROUPS[r] == grp), key=lambda r: (rel_n[r], r))
        cand = by_rel[rel]
        best = max(range(len(cand)),
                   key=lambda i: (-sum(seen[f] for f in feats(cand[i])), bool(cand[i]["distractor_ids"])))
        q = cand.pop(best)
        out.append(q)
        rel_n[rel] += 1
        grp_n[grp] += 1
        seen.update(feats(q))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", type=Path, default=Path("/mnt/Backup2nd/Dataset/hm3d_val"))
    p.add_argument("--out_dir", type=Path, default=Path("/mnt/Backup2nd/Dataset/hm3d_val/iref_style_queries"))
    p.add_argument("--scenes", nargs="*", default=SCENES)
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for scene in args.scenes:
        rng = random.Random(args.seed)
        info = json.loads((args.data_root / scene / "scene_info.json").read_text())
        qs = generate(info, rng)
        sub = sample(qs, args.n, rng)
        (args.out_dir / f"{scene}_all.json").write_text(json.dumps(qs))
        (args.out_dir / f"{scene}_sampled{args.n}.json").write_text(json.dumps(sub, indent=1))
        print(f"{scene}: all={len(qs)} sampled={len(sub)} rooms={len({q['region_id'] for q in sub})} "
              f"target_cls={len({q['target_class'] for q in sub})} "
              f"with_distractors={sum(bool(q['distractor_ids']) for q in sub)} "
              f"group={dict(sorted(Counter(GROUPS[q['relation']] for q in sub).items()))}")


if __name__ == "__main__":
    main()
