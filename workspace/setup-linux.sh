#!/usr/bin/env bash
# Run inside the user's existing Ubuntu/WSL, not in native Windows.
set -euo pipefail
staging=${1:?Pass the Windows staging folder translated by wslpath}
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || { echo 'Requires x86_64 Linux'; exit 1; }
[[ -f "$staging/installers/SHA256SUMS" ]] || { echo 'Installer download/verification is incomplete'; exit 1; }
build_root="$HOME/ascend-model-build"
if [[ -d "$build_root" && ! -f "$build_root/.codex-ascend-managed" ]]; then
  echo "Existing unmanaged directory: $build_root. Preserve it and choose a separate build directory."
  exit 1
fi
mkdir -p "$build_root/logs" "$build_root/tools" "$build_root/smoke" "$staging/logs"
touch "$build_root/.codex-ascend-managed"
setup_log="$staging/logs/setup-linux-$(date +%Y%m%d-%H%M%S).log"
progress_file="$staging/logs/current-setup-state.txt"
status_file="$staging/environment-status.json"
setup_phase=starting
printf '{"environment_ready":false,"status":"setup_in_progress"}\n' > "$status_file"
set_phase() {
  setup_phase=$1
  printf 'stage=%s\nupdated_at=%s\nlog=%s\n' "$setup_phase" "$(date -Iseconds)" "$setup_log" > "$progress_file"
}
on_setup_exit() {
  local setup_rc=$?
  printf 'exit_code=%s\n' "$setup_rc" >> "$progress_file"
  if (( setup_rc != 0 )); then
    printf '{"environment_ready":false,"status":"failed","stage":"%s","exit_code":%s}\n' "$setup_phase" "$setup_rc" > "$status_file"
    echo "Setup failed at $setup_phase (exit $setup_rc). Log: $setup_log"
  fi
}
trap on_setup_exit EXIT
set_phase starting
exec > >(tee -a "$setup_log") 2>&1
echo "[WSL/Linux host] Build directory: $build_root"
echo "[WSL/Linux host] Models: $staging/models"
if [[ $(id -u) == 0 ]]; then
  privileged=()
else
  privileged=(sudo)
fi
set_phase ubuntu_packages
required_packages=(git podman ca-certificates curl uidmap fuse-overlayfs slirp4netns)
missing_packages=()
for package in "${required_packages[@]}"; do
  if [[ $(dpkg-query -W -f='${Status}' "$package" 2>/dev/null || true) != 'install ok installed' ]]; then
    missing_packages+=("$package")
  fi
done
if (( ${#missing_packages[@]} > 0 )); then
  "${privileged[@]}" apt-get update
  "${privileged[@]}" apt-get install -y "${missing_packages[@]}"
else
  echo 'Required Ubuntu packages are already installed; reusing them.'
fi
set_phase verify_installers
( cd "$staging/installers" && sed 's/\r$//' SHA256SUMS | sha256sum -c - )
tar -xzf "$staging/installers/pixi-x86_64-unknown-linux-musl.tar.gz" -C "$build_root/tools" pixi
chmod +x "$build_root/tools/pixi"
pixi="$build_root/tools/pixi"
"$pixi" --version
project="$build_root/qwen-atlas-slm"
commit=15209f299fa372cccbd2226114f61a2f69f39a5f
if [[ ! -e "$project" ]]; then
  git -c core.autocrlf=false clone "https://github.com/MST19711/qwen-atlas-slm.git" "$project"
  git -C "$project" checkout --detach "$commit"
fi
[[ $(git -C "$project" rev-parse HEAD) == "$commit" ]] || { echo 'Existing project commit differs'; exit 1; }
[[ -z $(git -C "$project" status --porcelain --untracked-files=no) ]] || { echo 'Project has tracked modifications; preserve and review them'; exit 1; }
cd "$project"
set_phase export_dependencies
"$pixi" install --locked
"$pixi" run --locked python - <<'PY'
import importlib.metadata
import torch
for package in ('torch', 'transformers', 'onnx', 'onnxruntime', 'numpy', 'safetensors', 'huggingface_hub', 'accelerate'):
    print(package, importlib.metadata.version(package))
assert torch.ones(2, 2, dtype=torch.float16).matmul(torch.ones(2, 2, dtype=torch.float16)).isfinite().all()
print('CPU FP16 basic tensor smoke check passed; no model export has run.')
PY
for package in Ascend-cann-toolkit_7.0.0_linux-x86_64.run Ascend-cann-kernels-310b-7.0.0-linux.noarch.rpm; do
  cp -u -- "$staging/installers/$package" "$project/docker/$package"
done
# WSL has no systemd user bus in this session. Explicit global options also
# propagate to the Buildah build path, unlike Podman's automatic fallback.
podman_cmd=(podman --cgroup-manager=cgroupfs --events-backend=file)
"${podman_cmd[@]}" info
image=localhost/cann-atc-rocky:v7
set_phase container_runtime_probe
"${podman_cmd[@]}" run --rm --cgroups=disabled --network=host docker.io/library/rockylinux:9 /bin/true
set_phase atc_container_build
"${podman_cmd[@]}" build --network=host -t "$image" -f docker/Containerfile.v2-cann7 docker/
exec bash "$staging/verify-compiler.sh" "$staging" "$build_root"
