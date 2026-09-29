"""Generate IRef-VLA-style, room-conditioned referential queries for HM3DSem walks.

Input : <data_root>/<split>/<scene>/scene_info.json (HOV-SG GT; Y-up, AABBs, region labels)
Output: <data_root>/<out_name>.json      self-describing benchmark file (sampled subset, all scenes)
        <data_root>/<out_name>_all.json  same format, every valid candidate
        <data_root>/<out_name>/<scene>_{all,sampled<N>}.json  raw per-scene lists (used by the viewer)

Every query has exactly one answer inside its room:
  * target/anchor classes exclude structure and vague labels
  * anchors are the only object of their class in the room
  * no same-class distractor satisfies a *relaxed* version of the relation
  * superlatives need a clear winner: >= MARGIN and >= RATIO x the runner-up distance, same winner
    by surface distance; for a same-class pair only one of closest/farthest is kept
Rooms labelled empty/unknown are skipped.

Cross-room tier (does the same statement also fit elsewhere in the scene?):
  T0 target class exists only in this room
  T1 target class exists in other rooms, statement does not hold there
  T2 statement also holds in a differently-labelled room -> the room label resolves it
  T3 statement also holds in a same-labelled room -> a room descriptor is appended
     ("the bedroom with the desk", "the bedroom next to the kitchen") that singles the room
     out among all same-labelled rooms; T3 queries without such a descriptor are dropped.
Sampling balances relation group x tier cells.
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
    "ceiling molding", "skirting board", "floor mat frame", "door knob",
    "air duct", "air vent", "vent", "ventilation", "ceiling vent", "ceiling duct", "ceiling dome",
    "bath wall", "compound wall", "fireplace wall", "recessed wall", "stair wall", "shower ceiling",
    "shower floor", "shower pipe", "shower door frame", "door/window frame", "frame", "mirror frame",
    "painting frame", "bedframe", "curtain rail", "panel", "paneling", "board", "doorstep", "step",
    "stair", "staircase handrail", "staircase trim",
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
# Room descriptors: salient objects only (no parts, wall art, or small clutter).
DESC_MIN_SIZE = 0.3  # m, largest AABB side
DESC_EXCLUDE = {
    "door", "window", "curtain", "blinds", "drawer", "cabinet door", "kitchen cabinet door",
    "kitchen cabinet drawer", "closet door", "door knob", "picture", "painting", "photo mount",
    "picture frame", "ceiling lamp", "wall lamp", "air vent", "fire alarm", "towel", "clothes",
    "pillow", "book", "toy", "plush toy", "box", "cardboard box", "bag", "shelf", "cabinet", "rack",
    "hat", "hanger", "clothes hanger", "towel bar", "shoe", "backpack",
}
DESC_EXCLUDE_SUBSTR = ("frame", "pipe", "rod", " with ", " of ")


def describable(cls):
    return cls not in DESC_EXCLUDE and cls not in EXCLUDE and not any(s in cls for s in DESC_EXCLUDE_SUBSTR)
ADJ_DIST = 0.15  # m, max floor-plan boundary distance for "next to" rooms
ADJ_PHRASES = ["next to", "adjacent to"]
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
MARGIN = 0.5  # m, superlative winner vs runner-up centre-distance gap
RATIO = 1.5  # superlative runner-up / winner distance ratio (farthest: winner / runner-up)


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
            wins = [(rel, superlative_winner(group, a, far)) for rel, (_, far) in ORDINAL.items()]
            wins = [(rel, w) for rel, w in wins if w is not None]
            if len(group) == 2 and len(wins) == 2:  # closest/farthest of a pair are mirror images
                wins = [rng.choice(wins)]
            for rel, win in wins:
                add(group[win], rel, [a])
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


def holds(room_cls, rel, tcls, acls):
    """Does '<tcls> <rel> <acls>' describe some object in a room given as {class: [Box]}?"""
    ts = room_cls.get(tcls, [])
    if not ts or not all(room_cls.get(a) for a in acls):
        return False
    if rel in BINARY:
        return any(BINARY[rel](t, a) for t in ts for a in room_cls[acls[0]])
    if rel == "between":
        return any(between(t, a1, a2) for t in ts for a1 in room_cls[acls[0]] for a2 in room_cls[acls[1]])
    # superlative: same test as in-room generation (a clear winner by MARGIN) for some anchor
    return any(superlative_winner(ts, a, ORDINAL[rel][1]) is not None for a in room_cls[acls[0]])


def superlative_winner(group, a, far):
    """Index of the object in `group` closest (far=False) / farthest to `a`, if it wins clearly:
    centre-distance gap >= MARGIN, runner-up/winner distance ratio >= RATIO, and the same winner
    under surface (AABB gap) distance."""
    if len(group) < 2:
        return None
    order = sorted(((dist(t, a), i) for i, t in enumerate(group)), reverse=far)
    (d1, win), (d2, _) = order[0], order[1]
    near_d, far_d = (d2, d1) if far else (d1, d2)
    if abs(d1 - d2) < MARGIN or far_d < RATIO * max(near_d, 1e-6):
        return None
    gaps = [gap(t, a) for t in group]
    surface_win = max(range(len(group)), key=gaps.__getitem__) if far else min(range(len(group)), key=gaps.__getitem__)
    return win if surface_win == win else None


def adjacency(regions):
    """Rooms on the same floor whose floor-plan boundaries come within ADJ_DIST."""
    from scipy.spatial import cKDTree

    pts = {r["id"]: np.asarray(r["bev_region_points"])[:, [0, 2]]
           for r in regions if len(r.get("bev_region_points") or []) > 0}
    trees = {k: cKDTree(v) for k, v in pts.items()}
    floor = {r["id"]: r.get("floor_id") for r in regions}
    adj = defaultdict(set)
    for a, b in combinations(sorted(pts), 2):
        if floor[a] == floor[b] and trees[b].query(pts[a], k=1)[0].min() <= ADJ_DIST:
            adj[a].add(b)
            adj[b].add(a)
    return adj


def room_descriptor(rid, label_of, room_cls, adj, used_cls, rng):
    """Phrase that singles room `rid` out among all rooms sharing its label, or None."""
    lab = label_of[rid]
    peers = [r for r in label_of if r != rid and label_of[r] == lab]
    options = {}

    salient = sorted(
        ((max(max(b.hi - b.lo) for b in bs), c) for c, bs in room_cls[rid].items()
         if describable(c) and c not in used_cls and max(max(b.hi - b.lo) for b in bs) >= DESC_MIN_SIZE),
        reverse=True)
    unique = [c for _, c in salient if not any(c in room_cls.get(p, {}) for p in peers)]
    if unique:  # among the most salient unique classes; sometimes name two
        k = 2 if len(unique) >= 2 and rng.random() < 0.3 else 1
        options["objects"] = rng.sample(unique[:4], k)
    else:  # no single class singles it out: try a pair
        for c1, c2 in combinations([c for _, c in salient], 2):
            if not any(c1 in room_cls.get(p, {}) and c2 in room_cls.get(p, {}) for p in peers):
                options["objects"] = [c1, c2]
                break

    nbr_labels = {label_of[n] for n in adj.get(rid, ()) if n in label_of} - {lab}
    nbr_labels = {l for l in nbr_labels if l.strip().lower() not in SKIP_ROOMS}
    good = sorted(l for l in nbr_labels
                  if not any(l in {label_of.get(n) for n in adj.get(p, ())} for p in peers))
    if good:
        options["adjacent"] = [rng.choice(good)]

    if not options:
        return None
    kind = rng.choice(sorted(options))
    if kind == "objects":
        cls = options["objects"]
        ids = [b.id for c in cls for b in room_cls[rid][c]]
        return {"type": "objects", "classes": cls, "object_ids": ids,
                "text": "with " + " and ".join(f"the {c}" for c in cls)}
    nlab = options["adjacent"][0]
    nids = [n for n in adj[rid] if label_of.get(n) == nlab]
    return {"type": "adjacent", "neighbor_label": nlab, "neighbor_region_ids": nids,
            "text": f"{rng.choice(ADJ_PHRASES)} the {nlab}"}


def generate(scene_info, rng):
    rooms = {r["id"]: r for r in scene_info["regions"]}
    label_of = {rid: r["category"] for rid, r in rooms.items()}
    room_cls = defaultdict(lambda: defaultdict(list))  # rid -> class -> [Box]
    objs = defaultdict(list)
    for o in scene_info["objects"]:
        if o["category"] not in EXCLUDE and max(o["aabb_dims"]) >= MIN_SIZE:
            b = Box(o)
            objs[o["region_id"]].append(b)
            room_cls[o["region_id"]][b.cls].append(b)
    adj = adjacency(scene_info["regions"])
    queries = []
    for rid in sorted(objs):
        room = rooms.get(rid)
        if room is None or room["category"].strip().lower() in SKIP_ROOMS:
            continue
        for t, rel, ancs, dis in region_queries(objs[rid], rng):
            acls = [a.cls for a in ancs]
            others = [r for r in room_cls if r != rid]
            rivals = [r for r in others if holds(room_cls[r], rel, t.cls, acls)]
            same_label = [r for r in rivals if label_of.get(r) == room["category"]]
            if same_label:
                tier = "T3"
            elif rivals:
                tier = "T2"
            elif any(room_cls[r].get(t.cls) for r in others):
                tier = "T1"
            else:
                tier = "T0"
            desc = None
            if tier == "T3":
                desc = room_descriptor(rid, label_of, room_cls, adj, {t.cls, *acls}, rng)
                if desc is None:
                    continue
            stmt = statement(t.cls, rng.choice(PHRASES[rel]), ancs)
            room_ref = room["category"] + (f" {desc['text']}" if desc else "")
            queries.append({
                "query": f"In the {room_ref}, {stmt}",
                "statement": stmt,
                "relation": rel,
                "relation_type": "ternary" if rel == "between" else "binary",
                "tier": tier,
                "region_id": rid,
                "region_label": room["category"],
                "room_descriptor": desc,
                "rival_rooms": [{"region_id": r, "label": label_of.get(r)} for r in rivals],
                "floor_id": room.get("floor_id"),
                "target_id": t.id,
                "target_class": t.cls,
                "target": box_json(t),
                "anchors": [box_json(a) for a in ancs],
                "anchor_classes": acls,
                "distractor_ids": [d.id for d in dis],
            })
    return queries


def sample(queries, n, rng):
    """Rotate tiers (least-picked first; four equal tiers put half the budget on cross-room T2/T3),
    then the least-picked relation group overall within that tier, then its least-picked relation;
    within it pick the candidate adding the most unseen target class / instance / anchor class /
    room, preferring ones with distractors."""
    by_key = defaultdict(list)  # (group, tier, relation) -> queries
    for q in queries:
        by_key[(GROUPS[q["relation"]], q["tier"], q["relation"])].append(q)
    for v in by_key.values():
        rng.shuffle(v)

    def feats(q):
        return ([("t", q["target_class"]), ("i", q["target_id"]), ("r", q["region_id"])]
                + [("a", a) for a in q["anchor_classes"]])

    seen, tier_n, grp_n, key_n, out = Counter(), Counter(), Counter(), Counter(), []
    while len(out) < n and any(by_key.values()):
        live = [k for k in by_key if by_key[k]]
        tier = min({k[1] for k in live}, key=lambda t: (tier_n[t], t))
        grp = min({k[0] for k in live if k[1] == tier}, key=lambda g: (grp_n[g], g))
        key = min((k for k in live if k[:2] == (grp, tier)), key=lambda k: (key_n[k], k))
        cand = by_key[key]
        best = max(range(len(cand)),
                   key=lambda i: (-sum(seen[f] for f in feats(cand[i])), bool(cand[i]["distractor_ids"])))
        q = cand.pop(best)
        out.append(q)
        tier_n[tier] += 1
        grp_n[grp] += 1
        key_n[key] += 1
        seen.update(feats(q))
    return out


SPEC = {
    "task": "3D referential grounding: given `query` and the scene's posed RGB-D walk (rgb/, depth/, pose/), "
            "return the single referred object (3D box or instance). Exactly one GT object per query.",
    "coordinate_frame": "Habitat world frame of scene_info.json: Y-up, metres; AABBs are world-axis-aligned.",
    "query_format": "\"In the <room label>[ <room descriptor>], the <target> that is/are <relation phrase> "
                    "the <anchor>[ and the <anchor2>]\"; `query_without_room` drops the room clause "
                    "(for room-conditioning ablations; it may be ambiguous by design for T2/T3).",
    "gt": "`gt_objects` follows hm3dsem/long_queries_obj_room.json (hier_id = floor_region_object, pcd relative "
          "to data_root). Anchors/distractors/room-descriptor objects are listed by object_id of the same scene_info.",
    "uniqueness": [
        "target and anchor classes exclude structure/parts/vague labels (see `excluded_classes`) and objects < 5 cm",
        "each anchor is the only object of its class in the room",
        "no other same-class object in the room satisfies a relaxed version of the relation",
        f"closest/farthest: winner beats runner-up by >= {MARGIN} m and >= {RATIO}x distance, same winner by "
        "surface (AABB-gap) distance",
        "rooms labelled empty/unknown are never used",
    ],
    "relations": {
        "near": "AABB gap <= 0.5 m (distractors must be > 1.0 m)",
        "above / below": "vertical gap 0.1-1.5 m and >= 30% horizontal footprint overlap",
        "on": "target bottom within 0.1 m of anchor top, >= 70% of target footprint over the anchor",
        "in": ">= 90% of target AABB volume inside anchor AABB, anchor >= 3x target volume",
        "between": "target centre projects to 20-80% of the anchor-anchor segment (floor plane), "
                   "offset <= min(0.25 x length, 1 m); anchors 0.5-5 m apart",
        "closest / farthest": "centre distance ranking among same-class objects in the room",
    },
    "relation_groups": {"proximity": ["near"], "vertical": ["above", "below"], "support": ["on"],
                        "containment": ["in"], "between": ["between"], "superlative": ["closest", "farthest"]},
    "tiers": {
        "T0": "target class occurs only in this room",
        "T1": "target class occurs in other rooms but the statement holds only here",
        "T2": "statement also holds in a room with a different label -> room label disambiguates",
        "T3": "statement also holds in a room with the same label -> `room_descriptor` disambiguates",
    },
    "room_descriptor": "T3 only. 'objects': salient class(es) present in this room and in no other same-label "
                       "room; 'adjacent': a room label that only this same-label room borders (floor-plan "
                       f"boundary <= {ADJ_DIST} m, same floor).",
    "rival_rooms": "other rooms (any label) where the room-less statement also holds",
    "suggested_evaluation": {
        "primary": "top-1 accuracy: predicted object == gt (instance id match), or 3D IoU >= 0.25 / 0.5 "
                   "against gt AABB when the method outputs boxes",
        "breakdowns": ["tier (T0-T3)", "relation_group", "room_descriptor.type", "has in-room distractor"],
        "room_ablation": "run `query` and `query_without_room`; the T2/T3 gap measures use of room information",
    },
}


def ply_points(path):
    """Vertex count from a PLY header (0 if missing)."""
    try:
        with open(path, "rb") as f:
            for line in f:
                if line.startswith(b"element vertex"):
                    return int(line.split()[-1])
                if line.startswith(b"end_header"):
                    break
    except FileNotFoundError:
        pass
    return 0


def gt_entry(o, rooms, rel_scene, root):
    pcd = f"{rel_scene}/objects/{o['id']}.ply"
    return {
        "floor_id": o.get("floor_id"), "region_id": o["region_id"], "object_id": o["id"],
        "hier_id": f"{o.get('floor_id')}_{o['region_id']}_{o['id']}", "category": o["category"],
        "room_category": rooms.get(o["region_id"], {}).get("category"),
        "aabb_center": o["aabb_center"], "aabb_dims": o["aabb_dims"],
        "obb_center": o.get("obb_center"), "obb_dims": o.get("obb_dims"), "obb_rotation": o.get("obb_rotation"),
        "n_points": ply_points(root / pcd), "pcd": pcd,
    }


def bundle(per_scene, root, split, n, seed, sampled):
    queries = []
    for scene, (info, qs) in per_scene.items():
        objs = {o["id"]: o for o in info["objects"]}
        rooms = {r["id"]: r for r in info["regions"]}
        rel_scene = f"{split}/{scene}"
        for i, q in enumerate(qs):
            stmt = q["statement"]
            queries.append({
                "id": f"{scene}/{i:04d}", "scene": scene, "query": q["query"],
                "query_without_room": stmt[0].upper() + stmt[1:],
                "relation": q["relation"], "relation_group": GROUPS[q["relation"]], "tier": q["tier"],
                "region_id": q["region_id"], "region_label": q["region_label"], "floor_id": q["floor_id"],
                "room_descriptor": q["room_descriptor"], "rival_rooms": q["rival_rooms"],
                "gt_objects": [gt_entry(objs[q["target_id"]], rooms, rel_scene, root)],
                "anchors": [{"object_id": a["id"], "category": a["class"]} for a in q["anchors"]],
                "distractor_ids": q["distractor_ids"],
            })
    c = lambda key: dict(sorted(Counter(q[key] for q in queries).items()))
    return {
        "name": "HM3DSem IRef-style room-conditioned referential queries",
        "subset": f"sampled {n}/scene (tier-balanced, then relation group)" if sampled else "all valid candidates",
        "split": split, "data_root": str(root), "scenes": list(per_scene),
        "scene_layout": {k: f"{split}/<scene>/{v}" for k, v in {
            "scene_info": "scene_info.json", "object_pcd": "objects/<object_id>.ply",
            "region_pcd": "regions/<region_id>.ply", "rgb": "rgb/", "depth": "depth/", "pose": "pose/"}.items()},
        "generator": {"script": "scripts/gen_hm3dsem_iref_queries.py (KeySG repo)", "seed": seed},
        "spec": SPEC,
        "excluded_classes": sorted(EXCLUDE),
        "num_queries": len(queries),
        "stats": {"per_scene": c("scene"), "tier": c("tier"), "relation": c("relation"),
                  "relation_group": c("relation_group"),
                  "with_in_room_distractor": sum(bool(q["distractor_ids"]) for q in queries),
                  "room_descriptor": dict(Counter(q["room_descriptor"]["type"] for q in queries
                                                  if q["room_descriptor"])),
                  "num_target_categories": len({q["gt_objects"][0]["category"] for q in queries})},
        "queries": queries,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", type=Path, default=Path("/mnt/Backup2nd/Dataset/hm3dsem"))
    p.add_argument("--split", default="val")
    p.add_argument("--out_name", default="iref_room_queries",
                   help="writes <data_root>/<out_name>.json (sampled), <out_name>_all.json, and per-scene "
                        "files under <data_root>/<out_name>/")
    p.add_argument("--scenes", nargs="*", default=SCENES)
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    per_dir = args.data_root / args.out_name
    per_dir.mkdir(parents=True, exist_ok=True)
    sampled, full = {}, {}
    for scene in args.scenes:
        rng = random.Random(args.seed)
        info = json.loads((args.data_root / args.split / scene / "scene_info.json").read_text())
        qs = generate(info, rng)
        sub = sample(qs, args.n, rng)
        full[scene], sampled[scene] = (info, qs), (info, sub)
        (per_dir / f"{scene}_all.json").write_text(json.dumps(qs))
        (per_dir / f"{scene}_sampled{args.n}.json").write_text(json.dumps(sub, indent=1))
        print(f"{scene}: all={len(qs)} sampled={len(sub)} rooms={len({q['region_id'] for q in sub})} "
              f"target_cls={len({q['target_class'] for q in sub})} "
              f"with_distractors={sum(bool(q['distractor_ids']) for q in sub)} "
              f"group={dict(sorted(Counter(GROUPS[q['relation']] for q in sub).items()))} "
              f"tier={dict(sorted(Counter(q['tier'] for q in sub).items()))} "
              f"all_tier={dict(sorted(Counter(q['tier'] for q in qs).items()))}")
    for name, data, is_sampled in [(args.out_name, sampled, True), (f"{args.out_name}_all", full, False)]:
        out = args.data_root / f"{name}.json"
        out.write_text(json.dumps(bundle(data, args.data_root, args.split, args.n, args.seed, is_sampled), indent=1))
        print("wrote", out)


if __name__ == "__main__":
    main()
