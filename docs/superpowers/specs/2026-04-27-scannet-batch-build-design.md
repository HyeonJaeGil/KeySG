# ScanNet Batch Build Script Design

## Goal

Add a repo-local helper script that runs `keysg-build` for multiple ScanNet scene IDs without changing the existing single-scene pipeline behavior.

## Scope

The script will:

- Accept a ScanNet scans root directory.
- Accept one or more `scene_id` values explicitly.
- Accept a base `output_dir`.
- Optionally print commands without executing them.

The script will not:

- Change `keysg-build` itself.
- Discover scene IDs automatically.
- Parallelize builds.

## Approach

Implement a Python CLI under `scripts/` to match the existing repo pattern. The script will construct one `keysg-build` command per scene:

`keysg-build dataset.kind=scannet dataset.root_dir=<scans_root>/<scene_id> output_dir=<base_output_dir>`

Execution will use `subprocess.run(..., check=True)` so failures stop the batch immediately. A `--dry-run` mode will print the generated commands for inspection and testing.

## Error Handling

- Reject empty scene lists at argument parsing time.
- Resolve `scans_root` and join per-scene directories deterministically.
- In execution mode, abort on the first non-zero subprocess exit.

## Testing

Add a unit test that invokes the script with `--dry-run` and verifies the emitted commands include the expected per-scene `dataset.root_dir` and shared `output_dir`.
