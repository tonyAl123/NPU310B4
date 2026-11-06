#!/usr/bin/env bash
set -euo pipefail
staging=$(cd -- "$(dirname -- "$0")" && pwd -P)
build_root="$HOME/ascend-model-build"
[[ -f "$staging/.configured/qwen35-ctx1024-img640.json" ]] || { echo 'First run tools/configure_request.py for profile qwen35-ctx1024-img640'; exit 1; }
pixi="$build_root/tools/pixi"
[[ -x "$pixi" ]] || { echo 'Compiler environment is missing'; exit 1; }
(cd -- "$staging/conversion-tools/qwen35a-mm1024-img640" && sha256sum -c SHA256SUMS)
exec "$pixi" run --locked --manifest-path "$build_root/qwen-atlas-slm/pixi.toml" python -u "$staging/conversion-tools/qwen35a-mm1024-img640/job.py" --staging "$staging"
