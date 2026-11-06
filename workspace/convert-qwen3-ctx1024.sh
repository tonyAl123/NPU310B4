#!/usr/bin/env bash
set -euo pipefail
staging=$(cd -- "$(dirname -- "$0")" && pwd -P)
build_root="$HOME/ascend-model-build"
[[ -f "$staging/.configured/qwen3-ctx1024.json" ]] || { echo 'First run tools/configure_request.py for profile qwen3-ctx1024'; exit 1; }
pixi="$build_root/tools/pixi"
[[ -x "$pixi" ]] || { echo 'Compiler environment is missing'; exit 1; }
exec "$pixi" run --locked --manifest-path "$build_root/qwen-atlas-slm/pixi.toml" python -u "$staging/conversion-tools/qwen3_1024_job.py" run --request "$staging/conversion-tools/prepared-requests/qwen3-06b-ctx1024-b75ea12b.json" --staging "$staging" --build-root "$build_root"
