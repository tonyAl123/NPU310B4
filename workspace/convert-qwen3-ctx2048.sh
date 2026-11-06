#!/usr/bin/env bash
set -euo pipefail
staging=$(cd -- "$(dirname -- "$0")" && pwd -P)
build_root="$HOME/ascend-model-build"
[[ -f "$staging/.configured/qwen3-ctx2048.json" ]] || { echo 'First run tools/configure_request.py for profile qwen3-ctx2048'; exit 1; }
pixi="$build_root/tools/pixi"
[[ -x "$pixi" ]] || { echo 'Compiler environment is missing'; exit 1; }
exec "$pixi" run --locked --manifest-path "$build_root/qwen-atlas-slm/pixi.toml" python -u "$staging/conversion-tools/qwen3_2048_job.py" run --request "$staging/conversion-tools/prepared-requests/qwen3-06b-ctx2048-4ab1b48f.json" --staging "$staging" --build-root "$build_root"
