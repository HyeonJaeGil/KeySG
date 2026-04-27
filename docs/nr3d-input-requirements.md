# NR3D Input Requirements

## What `nr3d_root` should contain

Think of `nr3d_root` as a folder with two things:

1. a file that says what people asked
2. a file that says where the GT objects are

Recommended layout:

```text
nr3d_root/
├── nr3d.csv
└── scene0011_00_gt.json
```

File names do not have to match exactly, but both data types must exist somewhere under the folder.

## File 1: query annotations

This file contains one row per language query.

Minimum fields:

- `scan_id` or `scene_id`
- `utterance`
- `target_id`

Example:

```json
{
  "scan_id": "scene0011_00",
  "utterance": "the chair near the desk",
  "target_id": "17"
}
```

Your current `nr3d.csv` is valid for this part.

## File 2: GT object boxes

This file contains one row per GT object.

Minimum fields:

- `scan_id` or `scene_id`
- `id` or `object_id`
- `bbox_center`
- `bbox_extent`

Example:

```json
{
  "scan_id": "scene0011_00",
  "id": "17",
  "bbox_center": [1.2, 0.8, -0.4],
  "bbox_extent": [0.6, 1.0, 0.7]
}
```

Without this file, evaluation cannot compute IoU.

## Current status

What you already have:

- `nr3d.csv`

What is still needed:

- a GT bbox file for `scene0011_00`

## Run

```bash
MPLCONFIGDIR=/tmp/matplotlib python scripts/nr3d_eval.py \
  --scene_dir output/keysg_rag1/ScanNet/scene0011_00 \
  --nr3d_root /path/to/nr3d_root
```
