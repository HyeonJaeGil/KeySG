# KeySG

[![Static Badge](https://img.shields.io/badge/-arXiv-B31B1B?logo=arxiv)](https://arxiv.org/abs/2510.01049)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository is the official implementation of the paper:

> **KeySG: Hierarchical Keyframe-Based 3D Scene Graphs**
>
> [Abdelrhman Werby](), [Dennis Rotondi](), [Fabio Scaparro](), and [Kai O. Arras](). <br>
>
> *arXiv preprint arXiv:2510.01049*, 2025 <br>
> (Accepted for *IEEE International Conference on Robotics and Automation (ICRA), Vienna, Austria*, 2026.)

<p align="center">
  <img src="docs/cover.png" alt="KeySG represents 3D indoor scenes as hierarchical graphs enriched with multi-modal context from keyframes, enabling scalable language-driven scene querying." width="800" />
</p>


---

## 🛠️ Installation

### Setup

```bash
# Clone
git clone https://github.com/keysg-lab/KeySG.git
cd keysg

# Conda environment
conda env create -f environment.yaml
conda activate keysg

# Install KeySG as a package (editable)
pip install -e .
```

### 🤖 Model Checkpoints

Download and place in `checkpoints/`:

```bash
mkdir -p checkpoints

# SAM 2.1 Large
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# RAM++
wget -P checkpoints https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth
```

### 🔑 API Keys

Create a `.env` file at the repo root:

```bash
OPENAI_API_KEY=sk-...   # Required for VLM descriptions and RAG queries
```

## 🗂️ Dataset Preparation

KeySG supports **ScanNet**, **Replica**, and **HM3DSem**. All three require posed RGB-D sequences as input. We follow the same preparation procedure as [HOV-SG](https://github.com/hovsg/HOV-SG) — please refer to their repository for full download and pre-processing instructions.

### ScanNet
Download from the [official ScanNet website](http://www.scan-net.org/) and extract `.sens` files using the [SensReader](https://github.com/ScanNet/ScanNet/tree/master/SensReader/python) tool. Each scene directory should contain RGB frames, depth frames, and camera poses.

### Replica
Download the scanned RGB-D trajectories from the [Nice-SLAM](https://github.com/cvg/nice-slam) project (not the original Replica dataset). The directory should contain `results/` with `frame*.jpg` / `depth*.png` and a `traj.txt` pose file.

### HM3DSem
Download `hm3d-val-habitat-v0.2.tar`, `hm3d-val-semantic-annots-v0.2.tar`, and `hm3d-val-semantic-configs-v0.2.tar` from [Matterport](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md). Then generate posed RGB-D sequences with the habitat-sim renderer — see [HOV-SG's gen_hm3dsem_walks_from_poses.py](https://github.com/hovsg/HOV-SG/blob/main/hovsg/data/hm3dsem/gen_hm3dsem_walks_from_poses.py).

---

## 🚀 Quick Start

### 1. 🏗️ Build a Scene Graph

```bash
# ScanNet scene
keysg-build dataset.kind=scannet dataset.root_dir=/data/ScanNet/scans/scene0011_00

# Replica scene
keysg-build dataset.kind=replica dataset.root_dir=/data/Replica/room0

# HM3DSem scene
keysg-build dataset.kind=hm3dsem dataset.root_dir=/data/HM3DSem/val/00824-Dd4bFSTQ8gi
```

Outputs land in `output/keysg_rag1/{Dataset}/{Scene}/` by default (configurable in `config/main_pipeline.yaml`).

You can also run the pipeline directly:

```bash
python main_pipeline.py dataset.kind=scannet dataset.root_dir=/data/ScanNet/scans/scene0011_00
```

### 2. 🎨 Visualize and Query

```bash
keysg-vis --scene_dir output/keysg_rag1/ScanNet/scene0011_00
# Open http://localhost:8080
```

The visualizer shows:
- **Floor / room / object point clouds** — per-instance colors, toggleable layers
- **Camera frustums** at each keyframe's world-space pose with RGB thumbnails
- **Object Grounding panel** — type a natural-language query, the matching object highlights in 3D
- **Open-Ended Q&A panel** — ask anything about the scene; the LLM answers with cited reasoning

Custom port:

```bash
keysg-vis --scene_dir <path> --port 8090
```

### 3. 🐍 Programmatic Access

```python
from keysg.graph import KeySGGraph

# Load scene graph (RAG index built from cache if available)
graph = KeySGGraph.from_output_dir("output/keysg_rag1/ScanNet/scene0011_00")

# --- Object grounding ---
result = graph.query("red chair near the window")
print(result.target_object.label, result.bbox_3d, result.confidence)

# --- Open-ended Q&A ---
answer = graph.answer_question("What appliances are in this kitchen?")
print(answer["answer"])
print(answer["reasoning"])
print(answer["relevant_object_ids"])

# --- Browse the hierarchy ---
for floor in graph.floors:
    print(f"Floor {floor.id}: {floor.summary}")
    for room in floor.rooms:
        print(f"  Room {room.id}: {len(room.objects)} objects, {len(room.keyframes)} keyframes")
```

### 4. 📏 Run Nr3D Evaluation

`scripts/nr3d_eval.py` evaluates one built ScanNet scene against Nr3D queries.

Prerequisites:

- a built KeySG scene output, for example `output/keysg_rag1/ScanNet/scene0011_00`
- a local `nr3d_data` directory containing:
  - Nr3D query annotations
    - preferred: `queries_by_scene/<scene>.json`, for example `queries_by_scene/scene0011_00.json`
    - fallback: a shared annotations file such as `nr3d.csv`
  - GT bbox files for the same scene, such as `scene0011_00_gt.json`
- `OPENAI_API_KEY` in `.env`, since query analysis and object selection use OpenAI models

Minimal example:

```bash
python scripts/nr3d_eval.py \
  --scene_dir output/keysg_rag1/ScanNet/scene0011_00
```

If Matplotlib warns about an unwritable config/cache directory in your environment, prepend `MPLCONFIGDIR=/tmp/matplotlib`.

Useful options:

```bash
# Evaluate only the first 10 queries
python scripts/nr3d_eval.py \
  --scene_dir output/keysg_rag1/ScanNet/scene0011_00 \
  --limit 10

# Save outputs to a custom directory
python scripts/nr3d_eval.py \
  --scene_dir output/keysg_rag1/ScanNet/scene0011_00 \
  --output_dir output/experiments/nr3d_eval_scene0011

# Save a second run in the same output directory without overwriting the first
python scripts/nr3d_eval.py \
  --scene_dir output/keysg_rag1/ScanNet/scene0011_00 \
  --output_dir output/experiments/nr3d_eval \
  --run_name with_frames \
  --include_frame_text

# Override the default local ./nr3d_data directory
python scripts/nr3d_eval.py \
  --scene_dir output/keysg_rag1/ScanNet/scene0011_00 \
  --nr3d_root /path/to/nr3d_root
```

Outputs:

- `<output_dir>/<scene>_<run_name>_results.json`
  Per-query grounding results, including predicted object id, predicted label, bbox, confidence, reason, retrieved candidates, retrieved frames, and strict IoU.
- `<output_dir>/<scene>_<run_name>_metrics.json`
  Aggregate metrics using strict axis-aligned IoU.
- `<output_dir>/<scene>_<run_name>_failed.json`
  Queries with strict IoU below `0.1`.
- `<output_dir>/<scene>_<run_name>_summary.txt`
  Human-readable metric summary.
- `<output_dir>/<scene>_<run_name>_debug.log`
  Per-query debug trace.
- `<output_dir>/<scene>_<run_name>_args.json`
  CLI arguments used for that run.
- `<output_dir>/<scene>_<run_name>_run_meta.json`
  Execution metadata for that run.

For the expected Nr3D folder contents, see [docs/nr3d-input-requirements.md](docs/nr3d-input-requirements.md).

---

## ⚙️ Configuration

All settings live in `config/main_pipeline.yaml`. Any key can be overridden from the CLI using Hydra syntax (`key=value`).

```yaml
dataset:
  kind: scannet               # scannet | hm3dsem | replica
  root_dir: /path/to/scene
  depth_scale: 1000.0
  depth_min: 0.3
  depth_max: 4.0

output_dir: output/keysg_rag1

load:
  scene_segmentation: false   # true = skip re-segmentation
  scene_description: false    # true = skip re-description
  nodes: false                # true = skip re-extraction

vlm:
  provider: openai            # openai | ollama
  model: gpt-5-mini

segmentation:
  fuse_every_k: 10
  voxel_size: 0.05

nodes:
  segmentor: gsam2            # segmentation backend
  object_tags: vlm            # vlm | ram
  use_keyframes_only: true    # run detection only on sampled keyframes
  skip_frames: 15

build_rag: false              # build RAG index at end of pipeline
```

### ⏩ Resume from partial results

```bash
# Skip re-segmentation and re-description, re-run node extraction
keysg-build \
  dataset.root_dir=/data/ScanNet/scans/scene0011_00 \
  load.scene_segmentation=true \
  load.scene_description=true \
  load.nodes=false
```

---

## 📂 Output Structure

```
output/{run_name}/{Dataset}/{Scene}/
├── config.yaml                       # Copy of run config
├── floor_summaries.json              # Floor-level text summaries
├── keysg_graph.json                  # Scene graph metadata
├── scene_description_index.json      # Room description index
├── hovfun.log                        # Run log
├── rag_cache/                        # Cached embeddings & FAISS indices
│   ├── graph_chunks_meta.json
│   ├── graph_embeddings.npy
│   ├── graph_faiss.index
│   ├── graph_frame_visual_*.{npy,index}
│   └── graph_object_visual_*.{npy,index}
└── segmentation/
    └── floor_{id}/
        └── room_{fid}_{rid}/
            ├── {rid}.pkl             # Room geometry
            ├── {rid}.pcd             # Room point cloud
            ├── room_{rid}_vlm.json   # VLM descriptions + keyframe data
            ├── keyframe_poses.json   # Camera poses for visualizer
            ├── keyframes/            # Saved keyframe RGB images
            ├── nodes/                # Extracted object nodes (*.pkl)
            └── labeled_keyframes/    # Keyframes annotated with object IDs
```


## 📄 Abstract

In recent years, 3D scene graphs have emerged as a powerful world representation, offering both geometric accuracy and semantic richness. Combining 3D scene graphs with large language models enables robots to reason, plan, and navigate in complex human-centered environments. However, current approaches for constructing 3D scene graphs are semantically limited to a predefined set of relationships, and their serialization in large environments can easily exceed an LLM’s context window. We introduce KeySG, a framework that represents 3D scenes as a hierarchical graph consisting of floors, rooms, objects, and functional elements, where nodes are augmented with multi-modal information extracted from keyframes selected to optimize geometric and visual coverage. The keyframes allow us to efficiently leverage VLMs to extract scene information, alleviating the need to explicitly model relationship edges between objects, enabling more general, task-agnostic reasoning and planning. Our approach can process complex and ambiguous queries while mitigating the scalability issues associated with large scene graphs by utilizing a hierarchical multi-modal retrieval-augmented generation (RAG) pipeline to extract relevant context from the graph. Evaluated across three distinct benchmarks, 3D object semantic segmentation, functional element segmentation, and complex query retrieval KeySG outperforms prior approaches on most metrics, demonstrating its superior semantic richness and efficiency.


## 📝 Citation

```bibtex
@article{werby2025keysg,
  title={KeySG: Hierarchical Keyframe-Based 3D Scene Graphs},
  author={Werby, Abdelrhman and Rotondi, Dennis and Scaparro, Fabio and Arras, Kai O.},
  journal={arXiv preprint arXiv:2510.01049},
  year={2025}
}
```
