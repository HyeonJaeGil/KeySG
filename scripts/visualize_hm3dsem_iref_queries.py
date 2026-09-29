"""Viser viewer for IRef-style HM3DSem queries (see gen_hm3dsem_iref_queries.py).

One port, scene dropdown + query dropdown. Selecting a query shows only its room's RGB point
cloud with target (green) / anchor (blue) / distractor (red) AABBs; "Clear" restores the full scene.

    python scripts/visualize_hm3dsem_iref_queries.py --port 8080
"""
import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import viser

SCENES = ["00824-Dd4bFSTQ8gi", "00829-QaLdnwvtxbs", "00843-DYehNKdT76V", "00861-GLAQ4DNUx5U",
          "00862-LT9Jq6dN3Ea", "00873-bxsVRursffK", "00877-4ok3usBNeis", "00890-6s7QHgap2fW"]
NONE = "(none)"
COLORS = {"target": (40, 220, 60), "anchor": (40, 120, 255), "distractor": (240, 50, 50)}


class Viewer:
    def __init__(self, args):
        self.args = args
        self.server = viser.ViserServer(host=args.host, port=args.port)
        self.server.scene.set_up_direction("+y")
        self.lock = threading.Lock()
        self.cache = {}  # ply path -> (points, colors)
        self.queries = []
        self.scene = None
        self.full = None  # full-scene point cloud handle
        self.focus = []  # handles of the current query view
        self._build_gui()
        self.load_scene(SCENES[0])

    # ---------- data ----------
    def cloud(self, path):
        if path not in self.cache:
            pc = o3d.io.read_point_cloud(str(path))
            pts, cols = np.asarray(pc.points, np.float32), np.asarray(pc.colors, np.float32)
            if len(pts) > self.args.max_points:  # ponytail: random subsample, voxel-downsample if detail matters
                idx = np.random.default_rng(0).choice(len(pts), self.args.max_points, replace=False)
                pts, cols = pts[idx], cols[idx]
            self.cache[path] = (pts, (cols * 255).astype(np.uint8))
        return self.cache[path]

    def label(self, i, q):
        return f"{i:03d} | {q['query']}"

    # ---------- gui ----------
    def _build_gui(self):
        gui = self.server.gui
        self.dd_scene = gui.add_dropdown("Scene", SCENES, initial_value=SCENES[0])
        self.dd_query = gui.add_dropdown("Query", [NONE], initial_value=NONE)
        with gui.add_folder("Navigate"):
            b_prev, b_next = gui.add_button("Prev query"), gui.add_button("Next query")
            b_clear = gui.add_button("Clear (show full scene)")
        with gui.add_folder("Display"):
            self.sl_size = gui.add_slider("Point size", min=0.001, max=0.05, step=0.001,
                                          initial_value=self.args.point_size)
            self.cb_dis = gui.add_checkbox("Show distractors", initial_value=True)
            self.cb_labels = gui.add_checkbox("Show box labels", initial_value=True)
        self.md = gui.add_markdown("")

        @self.dd_scene.on_update
        def _(_):
            self.load_scene(self.dd_scene.value)

        @self.dd_query.on_update
        def _(_):
            self.show_query(self.dd_query.value)

        @b_clear.on_click
        def _(_):
            self.dd_query.value = NONE
            self.show_query(NONE)

        def step(d):
            opts = list(self.dd_query.options)
            i = opts.index(self.dd_query.value) if self.dd_query.value in opts else 0
            self.dd_query.value = opts[max(1, min(len(opts) - 1, i + d))] if len(opts) > 1 else NONE

        b_prev.on_click(lambda _: step(-1))
        b_next.on_click(lambda _: step(+1))

        @self.sl_size.on_update
        def _(_):
            for h in [self.full, *self.focus]:
                if isinstance(h, viser.PointCloudHandle):
                    h.point_size = self.sl_size.value

        for cb in (self.cb_dis, self.cb_labels):
            cb.on_update(lambda _: self.show_query(self.dd_query.value))

    # ---------- scene ----------
    def clear_focus(self):
        for h in self.focus:
            h.remove()
        self.focus = []

    def load_scene(self, scene):
        with self.lock:
            if scene == self.scene:
                return
            self.scene = scene
            self.md.content = f"Loading **{scene}** ..."
            self.clear_focus()
            if self.full is not None:
                self.full.remove()
            pts, cols = self.cloud(self.args.data_root / scene / "scene_rgb.ply")
            self.full = self.server.scene.add_point_cloud(
                "/scene", pts, cols, point_size=self.sl_size.value, point_shading="flat")
            qf = self.args.query_dir / f"{scene}_{self.args.suffix}.json"
            self.queries = json.loads(qf.read_text()) if qf.exists() else []
            self.by_label = {self.label(i, q): q for i, q in enumerate(self.queries)}
        self.dd_query.options = [NONE, *self.by_label]
        self.dd_query.value = NONE
        self.show_query(NONE)

    def add_box(self, name, b, kind):
        c = COLORS[kind]
        self.focus.append(self.server.scene.add_box(
            f"/focus/{name}", color=c, dimensions=tuple(b["dims"]), position=tuple(b["center"]),
            wireframe=True))
        if self.cb_labels.value:
            top = np.asarray(b["center"]) + [0, b["dims"][1] / 2 + 0.05, 0]
            self.focus.append(self.server.scene.add_label(
                f"/focus/{name}_label", f"{kind}: {b['class']} ({b['id']})", position=tuple(top)))

    def show_query(self, value):
        with self.lock:
            self.clear_focus()
            q = self.by_label.get(value)
            if q is None:
                if self.full is not None:
                    self.full.visible = True
                self.md.content = (f"**{self.scene}** — {len(self.queries)} queries  \n"
                                   "Full scene shown. Pick a query to focus its room.")
                return
            self.full.visible = False
            pts, cols = self.cloud(self.args.data_root / self.scene / "regions" / f"{q['region_id']}.ply")
            self.focus.append(self.server.scene.add_point_cloud(
                "/focus/region", pts, cols, point_size=self.sl_size.value, point_shading="flat"))
            self.add_box("target", q["target"], "target")
            for i, a in enumerate(q["anchors"]):
                self.add_box(f"anchor{i}", a, "anchor")
            if self.cb_dis.value:
                objs = self.objects(self.scene)
                for d in q["distractor_ids"]:
                    o = objs[d]
                    self.add_box(f"dis{d}", {"id": d, "class": o["category"], "center": o["aabb_center"],
                                             "dims": o["aabb_dims"]}, "distractor")
            self.md.content = (
                f"**{q['query']}**  \n"
                f"relation: `{q['relation']}` · room: {q['region_label']} (region {q['region_id']}, "
                f"floor {q['floor_id']})  \n"
                f"<span style='color:#28dc3c'>■</span> target: {q['target_class']} ({q['target_id']})  \n"
                f"<span style='color:#2878ff'>■</span> anchor: "
                + ", ".join(f"{a['class']} ({a['id']})" for a in q["anchors"]) + "  \n"
                f"<span style='color:#f03232'>■</span> distractors: {len(q['distractor_ids'])}")

    def objects(self, scene):
        key = ("objects", scene)
        if key not in self.cache:
            info = json.loads((self.args.data_root / scene / "scene_info.json").read_text())
            self.cache[key] = {o["id"]: o for o in info["objects"]}
        return self.cache[key]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", type=Path, default=Path("/mnt/Backup2nd/Dataset/hm3d_val"))
    p.add_argument("--query_dir", type=Path, default=Path("/mnt/Backup2nd/Dataset/hm3d_val/iref_style_queries"))
    p.add_argument("--suffix", default="sampled150", help="query file = <scene>_<suffix>.json (e.g. all)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--point_size", type=float, default=0.01)
    p.add_argument("--max_points", type=int, default=2_000_000)
    Viewer(p.parse_args())
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
