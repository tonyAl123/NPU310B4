#!/usr/bin/env bash
# Verify the already installed environment. No apt install or image rebuild.
set -euo pipefail
staging=${1:?Pass the Windows staging folder translated by wslpath}
build_root=${2:-"$HOME/ascend-model-build"}
pixi="$build_root/tools/pixi"
project="$build_root/qwen-atlas-slm"
image=localhost/cann-atc-rocky:v7
[[ -x "$pixi" && -d "$project" ]] || { echo 'The export environment is not installed'; exit 1; }
[[ -f "$build_root/.codex-ascend-managed" ]] || { echo 'Not a managed build directory'; exit 1; }
run_id="verify-$(date +%Y%m%d-%H%M%S)-$$"
smoke_dir="$build_root/smoke/$run_id"
shared_smoke="$staging/builds/environment-smoke/$run_id"
mkdir -p "$smoke_dir" "$shared_smoke" "$staging/logs"
setup_log="$staging/logs/$run_id.log"
progress_file="$staging/logs/current-setup-state.txt"
status_file="$staging/environment-status.json"
setup_phase=verify_existing_environment
printf '{"environment_ready":false,"status":"verification_in_progress"}\n' > "$status_file"
set_phase() {
  setup_phase=$1
  printf 'stage=%s\nupdated_at=%s\nlog=%s\n' "$setup_phase" "$(date -Iseconds)" "$setup_log" > "$progress_file"
}
on_verify_exit() {
  local verify_rc=$?
  printf 'exit_code=%s\n' "$verify_rc" >> "$progress_file"
  if (( verify_rc != 0 )); then
    printf '{"environment_ready":false,"status":"failed","stage":"%s","exit_code":%s}\n' "$setup_phase" "$verify_rc" > "$status_file"
    echo "Verification failed at $setup_phase (exit $verify_rc). Log: $setup_log"
  fi
}
trap on_verify_exit EXIT
set_phase verify_existing_environment
exec > >(tee -a "$setup_log") 2>&1
cd "$project"
[[ $(git rev-parse HEAD) == 15209f299fa372cccbd2226114f61a2f69f39a5f ]] || { echo 'Project revision differs'; exit 1; }
podman_cmd=(podman --cgroup-manager=cgroupfs --events-backend=file)
"${podman_cmd[@]}" image exists "$image"
"${podman_cmd[@]}" image inspect "$image" > "$staging/logs/atc-image.json"
set_phase atc_tool_probe
# CANN 7 ATC has no --version flag. Record vendor installation metadata and help.
"${podman_cmd[@]}" run --rm --cgroups=disabled "$image" bash -c \
  'set -e; command -v atc; cat /etc/Ascend/ascend_cann_install.info; atc --help' \
  > "$staging/logs/atc-toolchain-info.log" 2>&1
echo 'ATC executable/help and CANN installation metadata recorded.'
export ASCEND_VERIFY_SMOKE="$smoke_dir"
set_phase onnx_reference_smoke
"$pixi" run --locked python - <<'PY'
import os
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
path = Path(os.environ['ASCEND_VERIFY_SMOKE'])
x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 4])
y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 4])
c = helper.make_tensor('c', TensorProto.FLOAT, [1, 4], [1.0] * 4)
graph = helper.make_graph([helper.make_node('Add', ['x', 'c'], ['y'])], 'setup_smoke', [x], [y], [c])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)], ir_version=7)
onnx.checker.check_model(model)
onnx.save(model, path / 'add.onnx')
session = ort.InferenceSession(str(path / 'add.onnx'), providers=['CPUExecutionProvider'])
actual = session.run(None, {'x': np.ones((1, 4), dtype=np.float32)})[0]
assert np.isfinite(actual).all()
np.testing.assert_allclose(actual, np.full((1, 4), 2.0, dtype=np.float32))
print('Synthetic ONNX checker and CPU reference execution passed.')
PY
set_phase atc_smoke_compile
echo '[ATC container] Compiling a synthetic Add graph for Ascend310B4.'
"${podman_cmd[@]}" run --rm --cgroups=disabled --network=host \
  -e TE_PARALLEL_COMPILER=1 -e MAX_COMPILE_CORE_NUMBER=1 \
  -v "$smoke_dir:/workspace:Z" -w /workspace \
  "$image" atc --model=/workspace/add.onnx --framework=5 \
  --output=/workspace/add --input_format=ND --input_shape='x:1,4' \
  --soc_version=Ascend310B4 --precision_mode=allow_fp32_to_fp16 --log=info \
  2>&1 | tee "$smoke_dir/atc.log"
[[ -s "$smoke_dir/add.om" ]] || { echo 'ATC produced no nonempty OM'; exit 1; }
grep -Eiq 'ATC.*(success|succeed)' "$smoke_dir/atc.log" || { echo 'No ATC final success message found; inspect the log'; exit 1; }
cp -- "$smoke_dir/add.onnx" "$smoke_dir/add.om" "$smoke_dir/atc.log" "$shared_smoke/"
sha256sum "$shared_smoke/add.onnx" "$shared_smoke/add.om"
export ASCEND_VERIFY_ROOT="$build_root"
export ASCEND_VERIFY_STAGE="$staging"
export ASCEND_VERIFY_LOG="$setup_log"
export ASCEND_VERIFY_SHARED="$shared_smoke"
"$pixi" run --locked python - <<'PY'
import hashlib
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path
stage = Path(os.environ['ASCEND_VERIFY_STAGE'])
artifacts = Path(os.environ['ASCEND_VERIFY_SHARED'])
image = json.loads((stage / 'logs/atc-image.json').read_text())
image_id = image[0].get('Id') or image[0].get('ID') or image[0].get('id')
assert image_id
records = []
for name in ('add.onnx', 'add.om', 'atc.log'):
    path = artifacts / name
    assert path.stat().st_size > 0
    with path.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(source.read()).hexdigest()
    records.append({'path': str(path.relative_to(stage)), 'bytes': path.stat().st_size, 'sha256': digest})
status = {
    'updated_at': datetime.now(timezone.utc).isoformat(),
    'environment_ready': True, 'status': 'ready', 'target_soc': 'Ascend310B4',
    'linux_build_root': os.environ['ASCEND_VERIFY_ROOT'],
    'models_linux_path': str(stage / 'models'),
    'project_commit': '15209f299fa372cccbd2226114f61a2f69f39a5f',
    'container_image': 'localhost/cann-atc-rocky:v7', 'container_image_id': image_id,
    'container_global_options': ['--cgroup-manager=cgroupfs', '--events-backend=file'],
    'container_run_options': ['--cgroups=disabled'],
    'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'onnx', 'onnxruntime', 'numpy', 'safetensors', 'huggingface_hub', 'accelerate')},
    'cann_install_metadata_log': 'logs/atc-toolchain-info.log',
    'onnx_checker': 'passed', 'onnx_cpu_reference': 'passed', 'smoke_compile': 'passed',
    'artifacts': records, 'model_export_tested': False, 'board_runtime_verified': False,
    'setup_log': os.environ['ASCEND_VERIFY_LOG'],
}
(stage / 'environment-status.json').write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
(artifacts / 'verification-manifest.json').write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
PY
set_phase complete
echo "Compiler environment verified. Models: $staging/models"
echo "Verification artifacts: $shared_smoke"
