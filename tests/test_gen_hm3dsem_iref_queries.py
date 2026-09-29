import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gen_hm3dsem_iref_queries as g  # noqa: E402


def obj(i, cls, center, dims, rid=0):
    return {"id": i, "category": cls, "region_id": rid, "aabb_center": center, "aabb_dims": dims}


def test_unique_answers_and_filters():
    info = {
        "regions": [{"id": 0, "category": "bedroom", "floor_id": 0},
                    {"id": 1, "category": "empty room", "floor_id": 0}],
        "objects": [
            obj(1, "bed", [0, 0.3, 0], [2, 0.6, 2]),
            obj(2, "pillow", [0, 0.7, 0], [0.5, 0.2, 0.4]),       # on the bed
            obj(3, "pillow", [4, 0.1, 4], [0.5, 0.2, 0.4]),       # far away on the floor
            obj(4, "lamp", [0, 2.2, 0], [0.4, 0.3, 0.4]),         # above the bed
            obj(5, "wall", [1.1, 1.2, 0], [0.1, 2.4, 3]),         # excluded as target/anchor
            obj(6, "chair", [0, 0.4, 0], [0.5, 0.8, 0.5], rid=1),  # room skipped
        ],
    }
    qs = g.generate(info, random.Random(0))
    assert qs and all(q["region_label"] == "bedroom" for q in qs)
    assert all("wall" not in [q["target_class"], *q["anchor_classes"]] for q in qs)
    on_bed = [q for q in qs if q["relation"] == "on" and q["anchor_classes"] == ["bed"]]
    assert [q["target_id"] for q in on_bed] == [2]
    assert on_bed[0]["distractor_ids"] == [3]
    assert on_bed[0]["query"].startswith("In the bedroom, the pillow that is ")
    assert any(q["relation"] == "above" and q["target_id"] == 4 for q in qs)
    # pillow can't be anchor (two of them)
    assert all("pillow" not in q["anchor_classes"] for q in qs)
    assert len(g.sample(qs, 3, random.Random(0))) == 3


def test_superlative_needs_clear_winner():
    def box(i, x):
        return g.Box(obj(i, "chair", [x, 0.4, 0], [0.4, 0.8, 0.4]))

    anchor = g.Box(obj(0, "table", [0, 0.4, 0], [0.2, 0.8, 0.2]))
    assert g.superlative_winner([box(1, 1.0), box(2, 1.2)], anchor, far=False) is None  # too close to call
    assert g.superlative_winner([box(1, 1.0), box(2, 2.0)], anchor, far=False) == 0
    assert g.superlative_winner([box(1, 1.0), box(2, 2.0)], anchor, far=True) == 1


def test_same_label_rival_gets_room_descriptor():
    def bev(x0):
        return [[x0, 0, 0], [x0 + 3, 0, 0], [x0 + 3, 0, 3], [x0, 0, 3]]

    info = {
        "regions": [{"id": 0, "category": "bedroom", "floor_id": 0, "bev_region_points": bev(0)},
                    {"id": 1, "category": "bedroom", "floor_id": 0, "bev_region_points": bev(10)},
                    {"id": 2, "category": "kitchen", "floor_id": 0, "bev_region_points": bev(13.1)}],
        "objects": [
            obj(1, "bed", [1, 0.3, 1], [2, 0.6, 2], rid=0),
            obj(2, "lamp", [1, 2.2, 1], [0.4, 0.3, 0.4], rid=0),
            obj(3, "desk", [2.5, 0.4, 2.5], [1.0, 0.8, 0.5], rid=0),
            obj(4, "bed", [11, 0.3, 1], [2, 0.6, 2], rid=1),
            obj(5, "lamp", [11, 2.2, 1], [0.4, 0.3, 0.4], rid=1),
        ],
    }
    qs = g.generate(info, random.Random(0))
    lamp0 = [q for q in qs if q["relation"] == "above" and q["target_id"] == 2]
    assert lamp0 and lamp0[0]["tier"] == "T3"
    assert lamp0[0]["room_descriptor"]["text"] == "with the desk"
    assert lamp0[0]["query"].startswith("In the bedroom with the desk, the lamp")
    # bedroom 1 is singled out only by its kitchen neighbour (bedroom 0 has none)
    lamp1 = [q for q in qs if q["relation"] == "above" and q["target_id"] == 5]
    assert lamp1 and lamp1[0]["room_descriptor"]["type"] == "adjacent"
    assert lamp1[0]["room_descriptor"]["neighbor_label"] == "kitchen"
    assert all(q["tier"] in {"T0", "T1", "T2", "T3"} for q in qs)
