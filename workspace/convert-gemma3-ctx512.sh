#!/usr/bin/env bash
set -euo pipefail
staging=$(cd -- "$(dirname -- "$0")" && pwd -P)
build_root="$HOME/ascend-model-build"
[[ -f "$staging/.configured/gemma3-ctx512.json" ]] || { echo 'First run tools/configure_request.py for profile gemma3-ctx512'; exit 1; }
pixi="$build_root/tools/pixi"
[[ -x "$pixi" ]] || { echo 'Compiler environment is missing'; exit 1; }
exec "$pixi" run --locked --manifest-path "$build_root/qwen-atlas-slm/pixi.toml" python -u "$staging/conversion-tools/gemma3_512_job.py" run --request "$staging/conversion-tools/prepared-requests/gemma3-1b-ctx512-befa3c1d.json" --staging "$staging" --build-root "$build_root"
