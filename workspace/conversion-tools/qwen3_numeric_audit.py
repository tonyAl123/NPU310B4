"""Diagnose an existing failed export without changing its acceptance gates."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import uuid


GRAPH_SHA = '076894dae3591efc336cd1b0d5f4ced40f9488ddfedac55b5456d8b8791f85a5'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(4 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def cache_slots(cache):
    import numpy as np
    return np.stack([np.stack([layer.keys[0, :, -1].detach().float().numpy(),
                               layer.values[0, :, -1].detach().float().numpy()])
                     for layer in cache.layers])


def execute_backend(args):
    import numpy as np
    local, output = Path(args.local), Path(args.output)
    request = read(local / 'request.json')
    cfg = request['config']
    count, heads, dim, length = cfg['num_hidden_layers'], cfg['num_key_value_heads'], cfg['head_dim'], request['context']
    names = [f'past_{kind}_{i}' for i in range(count) for kind in ('k', 'v')]
    is_ort = args.mode.startswith('ort-')
    is_wrapper = args.mode == 'wrapper-fp16'
    if is_ort:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 8, 1
        if args.mode == 'ort-disabled':
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(str(local / 'onnx/qwen3_06b_ctx256.onnx'),
                                       sess_options=options, providers=['CPUExecutionProvider'])
        outputs = ['logits'] + [name.replace('past_', 'present_') for name in names]
    else:
        import torch
        from transformers import AutoModelForCausalLM, DynamicCache
        torch.set_num_threads(8)
        dtype = torch.float16 if is_wrapper else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            str(local / 'model'), dtype=dtype, device_map='cpu', local_files_only=True,
            trust_remote_code=False, attn_implementation='eager', low_cpu_mem_usage=True).eval()
        assert all(p.dtype == dtype for p in model.parameters())
        if is_wrapper:
            spec = importlib.util.spec_from_file_location('frozen_exporter', local / 'export_qwen3_kvcache.py')
            exporter = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(exporter)
            exporter.Qwen3Attention.forward = exporter._patched_attention_forward
            exporter.Qwen3DecoderLayer.forward = exporter._patched_decoder_forward
            wrapper = exporter.KVCacheWrapper(model, length)

    for case in read(local / 'reference/cases.json'):
        with np.load(local / 'reference' / (case['name'] + '.npz'), allow_pickle=False) as reference:
            tokens = reference['tokens'].copy()
        initial = {name: np.zeros((1, heads, length, dim), dtype=np.float16) for name in names}
        if case.get('initial_cache'):
            with np.load(local / 'reference' / case['initial_cache'], allow_pickle=False) as seed:
                initial = {name: seed[name].copy() for name in names}
        if is_ort:
            cache = initial
        elif is_wrapper:
            cache = [torch.from_numpy(initial[name]) for name in names]
        else:
            cache = DynamicCache(config=model.config)
            # Boundary uses exactly the same FP16 prefix as the ONNX input,
            # cast to FP32, so the final-step comparison has identical inputs.
            if case.get('initial_cache'):
                position = case['start_position']
                for i in range(count):
                    k, v = [torch.from_numpy(initial[f'past_{kind}_{i}'][:, :, :position].copy()).to(dtype)
                            for kind in ('k', 'v')]
                    cache.update(k, v, i)
        logits, states = [], []
        for step, token in enumerate(tokens):
            position = case.get('start_position', 0) + step
            if is_ort:
                values = session.run(outputs, {'input_ids': np.array([[token]], dtype=np.int64),
                                               'position': np.array([position], dtype=np.int64), **cache})
                scores = values[0][0, -1].astype(np.float32)
                inserted = []
                for i in range(count):
                    pair = []
                    for j in range(2):
                        name = names[2 * i + j]
                        value = values[1 + 2 * i + j]
                        assert value.shape == (1, heads, length, dim) and np.isfinite(value).all()
                        assert np.array_equal(value[:, :, :position], cache[name][:, :, :position])
                        assert np.array_equal(value[:, :, position + 1:], cache[name][:, :, position + 1:])
                        pair.append(value[0, :, position].astype(np.float32))
                        cache[name] = value
                    inserted.append(pair)
                inserted = np.asarray(inserted)
            else:
                with torch.inference_mode():
                    if is_wrapper:
                        values = wrapper(torch.tensor([[token]], dtype=torch.int64),
                                         torch.tensor([position], dtype=torch.int64), *cache)
                        scores = values[0][0, -1].float().numpy()
                        updated = list(values[1:])
                        for before, after in zip(cache, updated):
                            assert torch.equal(before[:, :, :position], after[:, :, :position])
                            assert torch.equal(before[:, :, position + 1:], after[:, :, position + 1:])
                        cache = updated
                        inserted = np.stack([np.stack([cache[2*i+j][0, :, position].float().numpy()
                                                        for j in range(2)]) for i in range(count)])
                    else:
                        result = model(input_ids=torch.tensor([[token]], dtype=torch.int64),
                                       position_ids=torch.tensor([[position]], dtype=torch.int64),
                                       past_key_values=cache, use_cache=True)
                        cache = result.past_key_values
                        scores = result.logits[0, -1].float().numpy()
                        inserted = cache_slots(cache)
            assert np.isfinite(scores).all() and np.isfinite(inserted).all()
            logits.append(scores.copy())
            states.append(inserted.copy())
            if step % 8 == 0 or step + 1 == len(tokens):
                print(f'{args.mode} {case["name"]}: {step+1}/{len(tokens)}', flush=True)
        np.savez(output / (args.mode + '-' + case['name'] + '.npz'), tokens=tokens,
                 logits=np.stack(logits), slots=np.stack(states))


def metrics(actual, expected):
    import numpy as np
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    delta = a - b
    rmse = float(np.sqrt(np.mean(delta * delta)))
    rms = float(np.sqrt(np.mean(b * b)))
    denominator = float(np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()))
    coordinates = np.unravel_index(np.abs(delta).argmax(), delta.shape)
    av, bv = float(a[coordinates]), float(b[coordinates])
    fp16_value = np.float16(abs(bv))
    spacing = float(np.nextafter(fp16_value, np.float16(np.inf)) - fp16_value)
    return {'max_abs': float(np.max(np.abs(delta))), 'rmse': rmse,
            'relative_rms': rmse / rms if rms else None,
            'cosine': float(np.sum(a*b) / denominator) if denominator else None,
            'worst_index': [int(x) for x in coordinates],
            'actual_at_worst': av, 'expected_at_worst': bv,
            'fp16_spacing_at_expected': spacing,
            'worst_error_in_local_fp16_steps': abs(av-bv)/spacing if spacing else None}


def summarize(local, output):
    import numpy as np
    report = {'status': 'diagnostics_complete', 'conversion_accepted': False,
              'graph_sha256': GRAPH_SHA, 'original_gates_unchanged': True,
              'boundary_fp32_seed': 'same native-FP16 prefix cast to FP32', 'cases': []}
    backends = ['native-fp32', 'wrapper-fp16', 'ort-default', 'ort-disabled']
    for case in read(local / 'reference/cases.json'):
        name = case['name']
        data = {}
        for backend in ['original-fp16'] + backends:
            path = local / 'reference' / (name + '.npz') if backend == 'original-fp16' else output / (backend + '-' + name + '.npz')
            with np.load(path, allow_pickle=False) as saved:
                data[backend] = {key: saved[key].copy() for key in saved.files}
        comparisons = []
        pairs = [('wrapper-fp16', 'original-fp16'), ('ort-default', 'original-fp16'),
                 ('ort-disabled', 'original-fp16'), ('ort-default', 'wrapper-fp16'),
                 ('ort-disabled', 'wrapper-fp16')] + [(x, 'native-fp32') for x in ['original-fp16', 'wrapper-fp16', 'ort-default', 'ort-disabled']]
        for actual, expected in pairs:
            for step in range(len(data[actual]['tokens'])):
                comparisons.append({'actual': actual, 'expected': expected,
                                    'position': case.get('start_position', 0) + step,
                                    'logits': metrics(data[actual]['logits'][step], data[expected]['logits'][step]),
                                    'cache_slot': metrics(data[actual]['slots'][step], data[expected]['slots'][step])})
        greedy = {}
        for backend, values in data.items():
            offset = case.get('prompt_tokens', 1) - 1
            ids = values['logits'][offset:].argmax(axis=-1).tolist()
            greedy[backend] = {'tokens': ids, 'match_original': ids == data['original-fp16']['logits'][offset:].argmax(axis=-1).tolist()}
        report['cases'].append({'name': name, 'greedy': greedy, 'comparisons': comparisons})
    write(output / 'audit-report.json', report)
    lines = ['# Qwen3 数值诊断', '', '诊断完成；这不是数值验收通过，也没有启动 ATC。', '',
             '复用原始 ONNX；原失败记录、输入模型和预设阈值未修改。', '',
             '边界 FP32 测试使用相同的 FP16 前缀缓存转成 FP32；聊天测试各后端按同一 token 序列独立累积缓存。', '',
             '| Case | Actual | Reference | Max logit RMSE | Max cache abs | Greedy match |',
             '|---|---|---|---:|---:|---|']
    for case in report['cases']:
        pairs = sorted({(c['actual'], c['expected']) for c in case['comparisons']})
        for actual, expected in pairs:
            rows = [r for r in case['comparisons'] if (r['actual'],r['expected']) == (actual,expected)]
            lines.append(f'| {case["name"]} | {actual} | {expected} | {max(r["logits"]["rmse"] for r in rows):.8f} | {max(r["cache_slot"]["max_abs"] for r in rows):.8f} | {case["greedy"][actual]["match_original"]} |')
    lines.extend(['', '详细报告中的 cache_slot.worst_index 依次为 layer、K/V（0/1）、KV head、channel；logits.worst_index 为词表索引。',
                  '每个位置记录误差最大处的实际值、参考值、局部 FP16 间距与误差间距比。保留全部 logits 和新增 KV 槽供复查。'])
    (output / 'AUDIT_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


def run(args):
    shared = Path(args.shared).resolve()
    manifest = read(shared / 'build-manifest.json')
    local = Path(manifest['linux_work_directory']).resolve()
    assert local.is_relative_to((Path.home() / 'ascend-model-build/conversions').resolve())
    assert manifest['status'] == 'failed' and manifest['stage'] == 'numeric_validation'
    assert sha(local / 'onnx/qwen3_06b_ctx256.onnx') == GRAPH_SHA, 'Unexpected graph; audit is pinned to the failed attempt'
    request = read(local / 'request.json')
    for record in request['files']:
        assert sha(local / 'model' / record['name']) == record['sha256'], 'Model snapshot changed'
    assert sha(local / 'export_qwen3_kvcache.py') == manifest['source']['adapted_exporter_sha256']
    for record in manifest['artifacts']:
        if record['path'].startswith('reference/'):
            assert sha(local / record['path']) == record['sha256'], 'Saved reference changed'
    identifier = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6]
    output = shared / 'audits' / identifier
    output.mkdir(parents=True)
    driver = output / 'qwen3_numeric_audit.py'
    shutil.copy2(__file__, driver)
    provenance = {'status': 'running', 'diagnostics_only': True, 'graph_sha256': GRAPH_SHA,
                  'original_manifest_sha256': sha(shared / 'build-manifest.json'), 'script_sha256': sha(driver),
                  'original_validation_sha256': sha(shared / 'validation.json'), 'commands': []}
    write(output / 'audit-status.json', provenance)
    write(shared / 'latest-audit.json', {'directory': str(output), 'status': 'running'})
    os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
                       'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'PYTHONUNBUFFERED': '1'})
    try:
        for backend in ['native-fp32', 'wrapper-fp16', 'ort-default', 'ort-disabled']:
            command = [sys.executable, '-u', str(driver), backend, '--local', str(local), '--output', str(output)]
            provenance['commands'].append(command)
            write(output / 'audit-status.json', provenance)
            print('DIAGNOSTIC PHASE:', backend, flush=True)
            with (output / (backend + '.log')).open('w', encoding='utf-8') as log:
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, errors='replace') as process:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                        log.flush()
                    code = process.wait()
            if code:
                raise RuntimeError(f'{backend} diagnostics failed with exit code {code}')
        summarize(local, output)
        provenance['status'] = 'diagnostics_complete'
    except Exception:
        provenance['status'] = 'failed'
        provenance['error'] = traceback.format_exc()
        print(provenance['error'], flush=True)
    provenance['artifacts'] = [{'name':p.name,'sha256':sha(p),'bytes':p.stat().st_size}
                               for p in sorted(output.iterdir()) if p.is_file() and p.name != 'audit-status.json']
    write(output / 'audit-status.json', provenance)
    write(shared / 'latest-audit.json', {'directory': str(output), 'status': provenance['status']})
    print('Audit result:', provenance['status'], '\nShared output:', output, flush=True)
    return 0 if provenance['status'] == 'diagnostics_complete' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run', 'native-fp32', 'wrapper-fp16', 'ort-default', 'ort-disabled'])
    parser.add_argument('--shared')
    parser.add_argument('--local')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.mode == 'run':
        return run(args)
    execute_backend(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
