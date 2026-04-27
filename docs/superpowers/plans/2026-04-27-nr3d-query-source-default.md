# NR3D Query Source Default Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `nr3d_data/queries_by_scene` the default scene query source and add a CLI flag to opt into `queries_by_scene_filtered`.

**Architecture:** Keep the query file resolution logic centralized in `eval_helpers.py` and thread a single boolean through the evaluation CLI. Cover the behavior with focused unit tests for both the helper and the script entry point.

**Tech Stack:** Python standard library (`argparse`, `unittest`, `subprocess`, `tempfile`), existing `eval_helpers.py` and `scripts/nr3d_eval.py`

---

### Task 1: Add failing tests for the new default and CLI flag

**Files:**
- Modify: `tests/test_eval_helpers_annotations.py`
- Test: `tests/test_eval_helpers_annotations.py`

- [ ] **Step 1: Write the failing tests**

```python
    def test_prefers_unfiltered_scene_file_by_default(self) -> None:
        annotations = _load_scene_annotations(
            f"/unused/path/{scene_name}",
            nr3d_root=str(root),
        )
        self.assertEqual(annotations[0]["utterance"], "from unfiltered scene file")

    def test_prefers_filtered_scene_file_when_requested(self) -> None:
        annotations = _load_scene_annotations(
            f"/unused/path/{scene_name}",
            nr3d_root=str(root),
            use_filtered_queries=True,
        )
        self.assertEqual(annotations[0]["utterance"], "from filtered scene file")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_eval_helpers_annotations -v`
Expected: FAIL because the helper still prefers filtered queries first and has no opt-in parameter.

- [ ] **Step 3: Write minimal implementation**

```python
def _preferred_scene_annotations_file(
    root: str,
    scene_name: str,
    *,
    use_filtered_queries: bool = False,
) -> Optional[str]:
    primary_dir = "queries_by_scene_filtered" if use_filtered_queries else "queries_by_scene"
    secondary_dir = "queries_by_scene" if use_filtered_queries else "queries_by_scene_filtered"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_eval_helpers_annotations -v`
Expected: PASS

### Task 2: Add the CLI flag and document it

**Files:**
- Modify: `scripts/nr3d_eval.py`
- Modify: `README.md`
- Modify: `docs/nr3d-input-requirements.md`
- Test: `tests/test_eval_helpers_annotations.py`

- [ ] **Step 1: Write the failing CLI-oriented test**

```python
        with patch("scripts.nr3d_eval._load_scene_annotations") as load_annotations:
            nr3d_eval.main()
            self.assertEqual(load_annotations.call_args.kwargs["use_filtered_queries"], True)
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run: `python -m unittest tests.test_eval_helpers_annotations.Nr3dEvalCliTest -v`
Expected: FAIL because the CLI does not expose `--use_filtered_queries`.

- [ ] **Step 3: Write minimal implementation and docs update**

```python
    parser.add_argument(
        "--use_filtered_queries",
        action="store_true",
        help="Use queries_by_scene_filtered/<scene>.json instead of the default queries_by_scene/<scene>.json.",
    )
```

- [ ] **Step 4: Run the targeted test to verify it passes**

Run: `python -m unittest tests.test_eval_helpers_annotations.Nr3dEvalCliTest -v`
Expected: PASS

- [ ] **Step 5: Run the full targeted regression suite**

Run: `python -m unittest tests.test_eval_helpers_annotations -v`
Expected: PASS
