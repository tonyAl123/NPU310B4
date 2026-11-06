"""Local Qwen3 conversion job; reference and ONNX validation run in separate processes."""
import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import traceback
import uuid


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        while block := source.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def slots(cache):
    import numpy as np
    return np.stack([np.stack([
        layer.keys[0, :, -1, :].detach().float().cpu().numpy(),
        layer.values[0, :, -1, :].detach().float().cpu().numpy(),
    ]) for layer in cache.layers])


def reference(args, request):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(8)
    model_path = Path(args.local) / 'model'
    out_dir = Path(args.local) / 'reference'
    out_dir.mkdir()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), dtype=torch.float16, device_map='cpu', local_files_only=True,
        trust_remote_code=False, attn_implementation='eager', low_cpu_mem_usage=True,
    ).eval()
    assert all(p.dtype == torch.float16 for p in model.parameters())
    cases = []
    count = request['validation']['new_tokens']
    for index, prompt in enumerate(request['validation']['prompts']):
        text = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        prompt_ids = tokenizer.encode(text, add_special_tokens=False)
        assert 0 < len(prompt_ids) + count <= request['context']
        cache, generated, inputs, logits, states = None, [], [], [], []
        with torch.inference_mode():
            for position in range(len(prompt_ids) + count - 1):
                token = prompt_ids[position] if position < len(prompt_ids) else generated[position - len(prompt_ids)]
                result = model(input_ids=torch.tensor([[token]], dtype=torch.int64),
                               position_ids=torch.tensor([[position]], dtype=torch.int64),
                               past_key_values=cache, use_cache=True)
                cache = result.past_key_values
                scores = result.logits[0, -1].detach().float().cpu().numpy()
                assert np.isfinite(scores).all()
                inputs.append(token)
                logits.append(scores)
                states.append(slots(cache))
                if position >= len(prompt_ids) - 1:
                    generated.append(int(scores.argmax()))
        name = f'chat-{index}'
        np.savez(out_dir / (name + '.npz'), tokens=np.array(inputs, dtype=np.int64),
                 logits=np.stack(logits), slots=np.stack(states))
        case = {'name': name, 'prompt': prompt, 'prompt_tokens': len(prompt_ids),
                'steps': len(inputs), 'generated_tokens': generated,
                'generated_text': tokenizer.decode(generated, skip_special_tokens=True)}
        cases.append(case)
        print('Native reference:', json.dumps(case, ensure_ascii=False), flush=True)

    # Produce a valid 255-token cache in the unpatched native model. The ONNX
    # validator will consume its last free slot at position 255 independently.
    length = request['context']
    pattern = tokenizer.encode('This is a context boundary validation sequence. ', add_special_tokens=False)
    boundary_tokens = (pattern * ((length + len(pattern) - 1) // len(pattern)))[:length]
    with torch.inference_mode():
        prefix = model(input_ids=torch.tensor([boundary_tokens[:-1]], dtype=torch.int64),
                       use_cache=True, logits_to_keep=1).past_key_values
        initial = {}
        for i, layer in enumerate(prefix.layers):
            for label, tensor in (('k', layer.keys), ('v', layer.values)):
                assert tensor.shape[2] == length - 1
                value = np.zeros((1, model.config.num_key_value_heads, length, model.config.head_dim), dtype=np.float16)
                value[:, :, :-1, :] = tensor.detach().cpu().numpy()
                initial[f'past_{label}_{i}'] = value
        np.savez(out_dir / 'boundary-initial.npz', **initial)
        final = model(input_ids=torch.tensor([[boundary_tokens[-1]]], dtype=torch.int64),
                      position_ids=torch.tensor([[length - 1]], dtype=torch.int64),
                      past_key_values=prefix, use_cache=True)
        np.savez(out_dir / 'boundary.npz', tokens=np.array([boundary_tokens[-1]], dtype=np.int64),
                 logits=final.logits[0, -1].detach().float().cpu().numpy()[None, :],
                 slots=slots(final.past_key_values)[None, ...])
    cases.append({'name': 'boundary', 'steps': 1, 'start_position': length - 1,
                  'initial_cache': 'boundary-initial.npz'})
    write_json(out_dir / 'cases.json', cases)
    print('Native reference complete; exporter monkey-patches were never imported.', flush=True)


def error_metrics(actual, expected):
    import numpy as np
    a = np.asarray(actual, dtype=np.float64).ravel()
    b = np.asarray(expected, dtype=np.float64).ravel()
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    difference = a - b
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return {'max_abs': float(np.abs(difference).max()),
            'rmse': float(np.sqrt(np.mean(difference * difference))),
            'cosine': float(np.dot(a, b) / denominator) if denominator else float(np.array_equal(a, b))}


def validate(args, request):
    import numpy as np
    import onnx
    import onnxruntime as ort
    local = Path(args.local)
    shared = Path(args.shared)
    graph_path = local / 'onnx/qwen3_06b_ctx256.onnx'
    onnx.checker.check_model(str(graph_path))
    graph = onnx.load(str(graph_path), load_external_data=False)
    initializers = {item.name for item in graph.graph.initializer}
    input_shapes, interface = [], {'inputs': [], 'outputs': []}
    for item in graph.graph.input:
        if item.name in initializers:
            continue
        shape = [d.dim_value for d in item.type.tensor_type.shape.dim]
        assert all(d > 0 for d in shape), f'Dynamic input not supported by this static job: {item.name}'
        input_shapes.append(item.name + ':' + ','.join(map(str, shape)))
        interface['inputs'].append({'name': item.name, 'shape': shape,
                                    'dtype': onnx.TensorProto.DataType.Name(item.type.tensor_type.elem_type)})
    for item in graph.graph.output:
        interface['outputs'].append({'name': item.name,
                                     'dtype': onnx.TensorProto.DataType.Name(item.type.tensor_type.elem_type)})
    write_json(shared / 'interface.json', interface)
    (shared / 'input-shape.txt').write_text(';'.join(input_shapes) + '\n', encoding='utf-8')
    cfg = request['config']
    layers, heads, dim, length = cfg['num_hidden_layers'], cfg['num_key_value_heads'], cfg['head_dim'], request['context']
    expected_inputs = {'input_ids', 'position'} | {f'past_{kind}_{i}' for i in range(layers) for kind in ('k', 'v')}
    assert {x['name'] for x in interface['inputs']} == expected_inputs
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(graph_path), sess_options=options, providers=['CPUExecutionProvider'])
    output_names = ['logits'] + [f'present_{kind}_{i}' for i in range(layers) for kind in ('k', 'v')]
    assert set(output_names) == {item.name for item in session.get_outputs()}
    thresholds = request['validation']
    report = {'status': 'running', 'checker': 'passed', 'thresholds': thresholds, 'cases': [], 'failures': []}
    write_json(shared / 'validation.json', report)
    cases = load(local / 'reference/cases.json')
    for case in cases:
        arrays = np.load(local / 'reference' / (case['name'] + '.npz'), allow_pickle=False)
        if case.get('initial_cache'):
            initial = np.load(local / 'reference' / case['initial_cache'], allow_pickle=False)
            cache = {key: initial[key].copy() for key in initial.files}
        else:
            cache = {f'past_{kind}_{i}': np.zeros((1, heads, length, dim), dtype=np.float16)
                     for i in range(layers) for kind in ('k', 'v')}
        predicted, step_metrics, cache_metrics = [], [], []
        for step, token in enumerate(arrays['tokens']):
            position = case.get('start_position', 0) + step
            assert 0 <= position < length
            feed = {'input_ids': np.array([[int(token)]], dtype=np.int64),
                    'position': np.array([position], dtype=np.int64), **cache}
            outputs = session.run(output_names, feed)
            scores = outputs[0][0, -1].astype(np.float32)
            assert outputs[0].shape == (1, 1, cfg['vocab_size'])
            metric = error_metrics(scores, arrays['logits'][step])
            step_metrics.append(metric)
            if metric['max_abs'] > thresholds['logits_max_abs'] or metric['rmse'] > thresholds['logits_rmse'] or metric['cosine'] < thresholds['logits_cosine_min']:
                report['failures'].append({'case': case['name'], 'position': position, 'kind': 'logits', **metric})
            predicted.append(int(scores.argmax()))
            new_slots = []
            for i in range(layers):
                layer_slots = []
                for offset, kind in enumerate(('k', 'v')):
                    name = f'past_{kind}_{i}'
                    updated = outputs[1 + 2 * i + offset]
                    assert updated.shape == (1, heads, length, dim) and np.isfinite(updated).all()
                    assert np.array_equal(updated[:, :, :position, :], cache[name][:, :, :position, :]), f'Past cache overwritten: {name}, position {position}'
                    assert np.array_equal(updated[:, :, position + 1:, :], cache[name][:, :, position + 1:, :]), f'Future cache overwritten: {name}, position {position}'
                    layer_slots.append(updated[0, :, position, :].astype(np.float32))
                    cache[name] = updated
                new_slots.append(layer_slots)
            state_metric = error_metrics(np.array(new_slots), arrays['slots'][step])
            cache_metrics.append(state_metric)
            if state_metric['max_abs'] > thresholds['cache_slot_max_abs'] or state_metric['rmse'] > thresholds['cache_slot_rmse']:
                report['failures'].append({'case': case['name'], 'position': position, 'kind': 'cache_slot', **state_metric})
            if step % 8 == 0 or step + 1 == len(arrays['tokens']):
                print(f"ONNX {case['name']}: {step + 1}/{len(arrays['tokens'])}, position {position}, logits RMSE={metric['rmse']:.6f}", flush=True)
        result = {'name': case['name'], 'steps': len(predicted),
                  'logits_max_abs': max(x['max_abs'] for x in step_metrics),
                  'logits_max_rmse': max(x['rmse'] for x in step_metrics),
                  'logits_min_cosine': min(x['cosine'] for x in step_metrics),
                  'cache_slot_max_abs': max(x['max_abs'] for x in cache_metrics),
                  'cache_slot_max_rmse': max(x['rmse'] for x in cache_metrics),
                  'past_and_future_cache_preserved': True}
        if 'generated_tokens' in case:
            generated = predicted[case['prompt_tokens'] - 1:]
            result['greedy_generated_tokens'] = generated
            result['reference_generated_tokens'] = case['generated_tokens']
            result['greedy_match'] = generated == case['generated_tokens']
            if not result['greedy_match']:
                report['failures'].append({'case': case['name'], 'kind': 'greedy_generation_mismatch'})
        report['cases'].append(result)
        write_json(shared / 'validation.json', report)
        print('ONNX case result:', json.dumps(result), flush=True)
    report['status'] = 'passed' if not report['failures'] else 'failed'
    write_json(shared / 'validation.json', report)
    assert report['status'] == 'passed', 'Numerical validation failed; ATC compilation is blocked. See validation.json.'
    print('Independent multi-token, greedy prefill/decode and position-255 validation passed.', flush=True)


def run(args, request):
    staging = Path(args.staging).resolve()
    build_root = Path(args.build_root).resolve()
    job = Path(args.request).resolve().parent
    attempt_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6]
    shared = job / 'attempts' / attempt_id
    shared.mkdir(parents=True)
    local = build_root / 'conversions' / request['job_id'] / attempt_id
    local.mkdir(parents=True)
    for name in ('logs', 'source', 'deliverables'):
        (shared / name).mkdir()
    for name in ('onnx', 'om'):
        (local / name).mkdir()
    manifest = {'build_id': request['job_id'] + '/' + attempt_id,
                'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'running',
                'stage': 'preflight', 'request': request,
                'board_runtime_verified': False, 'stages': {}, 'commands': [],
                'linux_work_directory': str(local), 'artifacts': []}
    os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
                       'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8'})

    def phase(name):
        manifest['stage'] = name
        write_json(shared / 'build-manifest.json', manifest)
        write_json(job / 'conversion-status.json', {'status': manifest['status'], 'stage': name,
                                                   'attempt': str(shared.relative_to(staging)), 'local': str(local)})
        print('PHASE:', name, '| Windows-visible output:', shared, flush=True)

    def command(name, arguments, cwd=None):
        phase(name)
        manifest['commands'].append({'stage': name, 'argv': [str(a) for a in arguments], 'environment': 'ATC container via WSL host' if name == 'atc' else 'WSL/Linux host'})
        print('COMMAND:', shlex.join(map(str, arguments)), flush=True)
        with (shared / 'logs' / (name + '.log')).open('w', encoding='utf-8') as log:
            with subprocess.Popen([str(a) for a in arguments], cwd=cwd, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, errors='replace') as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                code = process.wait()
        manifest['stages'][name] = {'status': 'passed' if code == 0 else 'failed', 'exit_code': code, 'log': f'logs/{name}.log'}
        write_json(shared / 'build-manifest.json', manifest)
        if code:
            raise RuntimeError(f'{name} failed with exit code {code}')

    try:
        phase('preflight')
        environment = load(staging / 'environment-status.json')
        assert environment['environment_ready'] and environment['smoke_compile'] == 'passed'
        project = build_root / 'qwen-atlas-slm'
        assert subprocess.check_output(['git', '-C', str(project), 'rev-parse', 'HEAD'], text=True).strip() == request['project_commit']
        assert not subprocess.check_output(['git', '-C', str(project), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
        assert shutil.disk_usage(local).free > 12 * 1024**3, 'Need at least 12 GiB free for this build'
        available_ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_AVPHYS_PAGES')
        assert available_ram > 6 * 1024**3, 'Insufficient currently available RAM for the planned export'
        manifest['host'] = {'machine': os.uname().machine, 'kernel': os.uname().release,
                            'available_ram_at_start': available_ram, 'free_disk_at_start': shutil.disk_usage(local).free}
        manifest['versions'] = {name: importlib.metadata.version(name) for name in (
            'torch', 'transformers', 'onnx', 'onnxruntime', 'numpy', 'safetensors', 'huggingface_hub', 'accelerate')}
        phase('snapshot_model')
        model_source = (staging / request['model_relative_path']).resolve()
        assert model_source.is_relative_to((staging / 'models').resolve())
        snapshot = local / 'model'
        snapshot.mkdir()
        for record in request['files']:
            source = model_source / record['name']
            assert source.resolve().parent == model_source
            target = snapshot / record['name']
            shutil.copy2(source, target)
            assert target.stat().st_size == record['bytes'] and sha(target) == record['sha256'], f'Source changed or copy corrupted: {record["name"]}'
        original_path = project / 'scripts/export_qwen3_kvcache.py'
        original = original_path.read_text(encoding='utf-8')
        assert original.count('trust_remote_code=True') == 1
        adapted = original.replace('trust_remote_code=True', 'trust_remote_code=False, local_files_only=True, attn_implementation="eager"')
        adapted = adapted.replace('import torch\n', 'import torch\ntorch.set_num_threads(8)\n', 1)
        exporter = local / 'export_qwen3_kvcache.py'
        exporter.write_text(adapted, encoding='utf-8')
        (shared / 'source/export_qwen3_kvcache.py').write_text(adapted, encoding='utf-8')
        diff = ''.join(difflib.unified_diff(original.splitlines(True), adapted.splitlines(True),
                                            fromfile='upstream/export_qwen3_kvcache.py', tofile='build/export_qwen3_kvcache.py'))
        (shared / 'source/export.patch').write_text(diff, encoding='utf-8')
        driver = local / 'qwen3_job.py'
        shutil.copy2(Path(__file__), driver)
        shutil.copy2(driver, shared / 'source/qwen3_job.py')
        manifest['source'] = {'url': 'https://github.com/MST19711/qwen-atlas-slm', 'commit': request['project_commit'],
                              'original_exporter_sha256': sha(original_path), 'adapted_exporter_sha256': sha(exporter),
                              'lock_sha256': sha(project / 'pixi.lock'), 'job_driver_sha256': sha(driver),
                              'diff': 'source/export.patch'}
        write_json(local / 'request.json', request)
        common = ['--request', str(local / 'request.json'), '--local', str(local), '--shared', str(shared)]
        # The native reference is a fresh process and never imports the exporter.
        command('native_reference', [sys.executable, '-u', driver, 'reference', *common], cwd=project)
        command('export', [sys.executable, '-u', exporter, '--model-path', snapshot, '--max-len', str(request['context']),
                           '--output', local / 'onnx/qwen3_06b_ctx256.onnx'], cwd=project)
        command('numeric_validation', [sys.executable, '-u', driver, 'validate', *common], cwd=project)
        manifest['validation'] = load(shared / 'validation.json')
        assert manifest['validation']['status'] == 'passed'
        manifest['interface'] = load(shared / 'interface.json')
        shape = (shared / 'input-shape.txt').read_text(encoding='utf-8').strip()
        assert shape
        podman = ['podman', '--cgroup-manager=cgroupfs', '--events-backend=file']
        image = request['container_image_id']
        image_info = subprocess.check_output([*podman, 'image', 'inspect', image], text=True)
        write_json(shared / 'container-image.json', json.loads(image_info))
        (local / 'om/ascend-logs').mkdir()
        atc_args = ['--model=/model/qwen3_06b_ctx256.onnx', '--framework=5', '--output=/output/qwen3_06b_ctx256',
                    '--input_format=ND', '--input_shape=' + shape, '--soc_version=' + request['target_soc'],
                    '--precision_mode=allow_fp32_to_fp16', '--log=info']
        container = [*podman, 'run', '--rm', '--cgroups=disabled', '--network=host',
                     '-e', 'TE_PARALLEL_COMPILER=1', '-e', 'MAX_COMPILE_CORE_NUMBER=1',
                     '-e', 'ASCEND_PROCESS_LOG_PATH=/output/ascend-logs',
                     '-v', str(local / 'onnx') + ':/model:ro', '-v', str(local / 'om') + ':/output:Z', '-w', '/output',
                     image, 'bash', '-c',
                     'atc "$@"; result=$?; if [ -d /root/ascend/log ]; then cp -a /root/ascend/log /output/compiler-internal-logs || true; fi; exit "$result"',
                     'atc-job', *atc_args]
        manifest['compile'] = {'image_id': image, 'atc_arguments': atc_args, 'container_parallelism': 1}
        command('atc', container, cwd=local)
        om = local / 'om/qwen3_06b_ctx256.om'
        assert om.is_file() and om.stat().st_size > 0
        assert 'ATC run success' in (shared / 'logs/atc.log').read_text(encoding='utf-8')
        phase('deliver')
        shutil.copy2(om, shared / 'deliverables' / om.name)
        tokenizer = shared / 'deliverables/tokenizer'
        tokenizer.mkdir()
        for record in request['files']:
            if not record['name'].endswith(('.safetensors', '.bin', '.pt', '.pth')):
                shutil.copy2(snapshot / record['name'], tokenizer / record['name'])
        manifest['status'] = 'compiled'
        manifest['stages']['deliver'] = {'status': 'passed'}
    except Exception as error:
        manifest['status'] = 'failed'
        manifest['error'] = str(error)
        (shared / 'logs/failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        print(traceback.format_exc(), flush=True)
    finally:
        # Preserve generated graphs and every external data file even on failure.
        for name in ('onnx', 'reference'):
            if (local / name).is_dir() and any((local / name).iterdir()):
                shutil.copytree(local / name, shared / name)
        if (local / 'om').exists():
            for directory in ('ascend-logs', 'compiler-internal-logs'):
                if (local / 'om' / directory).exists():
                    shutil.copytree(local / 'om' / directory, shared / 'logs' / directory)
        for path in sorted(shared.rglob('*')):
            if path.is_file() and path.name not in ('build-manifest.json', 'BUILD_REPORT.md'):
                manifest['artifacts'].append({'path': str(path.relative_to(shared)), 'bytes': path.stat().st_size, 'sha256': sha(path)})
        manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
        if manifest['status'] == 'compiled':
            manifest['stage'] = 'complete'
        write_json(shared / 'build-manifest.json', manifest)
        write_json(shared / 'deliverables/build-manifest.json', manifest)
        report = f"# Qwen3 本地转换记录\n\n状态：{manifest['status']}\n\n目标：Ascend310B4，静态总上下文 256，CPU FP16 导出，ONNX opset 15。\n\n原始模型：{request['model_relative_path']}；上游 revision 未知，以输入文件 SHA256 固定本地版本。\n\nLinux 工作目录：{local}\n\n当前/最后阶段：{manifest['stage']}\n\n错误：{manifest.get('error', '无')}\n\n各阶段退出码、独立 PyTorch 参考与数值阈值、缓存验证、实际 ATC 参数及文件哈希见 build-manifest.json 与 validation.json。\n\n导出器修改只用于本次构建，关闭远程代码及联网加载并固定线程数，补丁见 source/export.patch。\n\nONNX 与全部外部数据保留在 onnx/。成功时 OM 和匹配的 tokenizer/config 位于 deliverables/。\n\n尚未执行开发板推理，board_runtime_verified=false；生成 OM 不代表板端兼容或性能已验证。\n"
        (shared / 'BUILD_REPORT.md').write_text(report, encoding='utf-8')
        (shared / 'deliverables/BUILD_REPORT.md').write_text(report, encoding='utf-8')
        phase('complete' if manifest['status'] == 'compiled' else manifest['stage'])
        print('FINAL STATUS:', manifest['status'], '| Output:', shared, flush=True)
    return 0 if manifest['status'] == 'compiled' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run', 'reference', 'validate'])
    parser.add_argument('--request', required=True)
    parser.add_argument('--local')
    parser.add_argument('--shared')
    parser.add_argument('--staging')
    parser.add_argument('--build-root')
    args = parser.parse_args()
    request = load(args.request)
    if args.mode == 'run':
        return run(args, request)
    if args.mode == 'reference':
        reference(args, request)
    else:
        validate(args, request)
    return 0


if __name__ == '__main__':
    sys.exit(main())
