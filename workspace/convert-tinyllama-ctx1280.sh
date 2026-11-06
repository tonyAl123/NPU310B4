#!/usr/bin/env bash
set -euo pipefail
staging=$(cd -- "$(dirname -- "$0")" && pwd -P)
build_root="$HOME/ascend-model-build"
[[ -f "$staging/.configured/tinyllama-ctx1280.json" ]] || { echo 'First run tools/configure_request.py for profile tinyllama-ctx1280'; exit 1; }
pixi="$build_root/tools/pixi"
[[ -x "$pixi" ]] || { echo 'Compiler environment is missing'; exit 1; }
package="$staging/conversion-tools/tinyllama-ctx1280"
(cd -- "$package" && sha256sum -c SHA256SUMS)
identifier=$(sha256sum "$package/request.json" | cut -c1-12)
workflow="$build_root/workflows/tinyllama-ctx1280-$identifier"
mkdir -p "$workflow"
cp -a -- "$package/." "$workflow/"
exec "$pixi" run --locked --manifest-path "$build_root/qwen-atlas-slm/pixi.toml" python -u "$workflow/tinyllama_job.py" run --request "$workflow/request.json" --staging "$staging" --build-root "$build_root"
