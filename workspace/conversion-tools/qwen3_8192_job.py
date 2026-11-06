"""8192-context Qwen3 job with isolated FP32 reference and chunked long-prefix probes."""
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
    import gc
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    torch.set_num_threads(8)
    local = Path(args.local)
    out_dir = local / 'reference'
    out_dir.mkdir()
    length = request['context']
    assert length == 8192 and length <= request['config']['max_position_embeddings']
    model = AutoModelForCausalLM.from_pretrained(
        str(local / 'model'), dtype=torch.float32, device_map='cpu', local_files_only=True,
        trust_remote_code=False, attn_implementation='eager', low_cpu_mem_usage=True).eval()
    assert all(p.dtype == torch.float32 for p in model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(str(local / 'model'),local_files_only=True,trust_remote_code=False)
    cases = []
    count = request['validation']['new_tokens']
    with torch.inference_mode():
        for index,prompt in enumerate(request['validation']['prompts']):
            text = tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,
                                                 add_generation_prompt=True,enable_thinking=False)
            prompt_ids = tokenizer.encode(text,add_special_tokens=False)
            assert 0 < len(prompt_ids)+count <= length
            cache,generated,inputs,logits,states = None,[],[],[],[]
            for position in range(len(prompt_ids)+count-1):
                token = prompt_ids[position] if position < len(prompt_ids) else generated[position-len(prompt_ids)]
                result = model(input_ids=torch.tensor([[token]],dtype=torch.int64),
                               position_ids=torch.tensor([[position]],dtype=torch.int64),past_key_values=cache,use_cache=True)
                cache = result.past_key_values
                scores = result.logits[0,-1].float().numpy().copy()
                assert np.isfinite(scores).all()
                inputs.append(token)
                logits.append(scores)
                states.append(slots(cache))
                if position >= len(prompt_ids)-1:
                    generated.append(int(scores.argmax()))
            name = f'chat-{index}'
            np.savez(out_dir / (name+'.npz'),tokens=np.array(inputs,dtype=np.int64),logits=np.stack(logits),slots=np.stack(states))
            case = {'name':name,'prompt':prompt,'prompt_tokens':len(prompt_ids),'steps':len(inputs),
                    'generated_tokens':generated,'generated_text':tokenizer.decode(generated,skip_special_tokens=True),
                    'reference_dtype':'float32','cache_mode':'independent growing native cache'}
            cases.append(case)
            print('Native FP32 reference:',json.dumps(case,ensure_ascii=False),flush=True)
            del cache,result,logits,states

        # Real causal prefix, in 64-token chunks. A full eager 8192-token
        # prefill would allocate quadratic attention tensors unnecessarily.
        pattern = tokenizer.encode('This is a context boundary validation sequence. ',add_special_tokens=False)
        boundary_tokens = (pattern*((length+len(pattern)-1)//len(pattern)))[:length]
        prefix_length = length-2
        cache = None
        for start in range(0,prefix_length,64):
            stop = min(start+64,prefix_length)
            prefill = model(input_ids=torch.tensor([boundary_tokens[start:stop]],dtype=torch.int64),
                            position_ids=torch.arange(start,stop,dtype=torch.int64)[None,:],
                            past_key_values=cache,use_cache=True,logits_to_keep=1)
            cache = prefill.past_key_values
            if stop % 256 == 0 or stop == prefix_length:
                print(f'Native FP32 long prefix: {stop}/{prefix_length} tokens (chunk 64)',flush=True)
        initial = {}
        for i,layer in enumerate(cache.layers):
            for kind,tensor in [('k',layer.keys),('v',layer.values)]:
                assert tensor.shape == (1,model.config.num_key_value_heads,prefix_length,model.config.head_dim)
                value = np.zeros((1,model.config.num_key_value_heads,length,model.config.head_dim),dtype=np.float16)
                value[:,:,:prefix_length] = tensor.numpy()
                assert np.isfinite(value).all()
                initial[f'past_{kind}_{i}'] = value
        del cache,prefill
        gc.collect()
        np.savez(out_dir / 'long-prefix-fp16.npz',**initial)
        np.save(out_dir / 'long-prefix-tokens.npy',np.array(boundary_tokens,dtype=np.int64))
        # Each position uses the same real prefix cast to FP16 for both
        # backends. This isolates decode/mask/rotary behavior at high positions.
        for position,steps in [(1023,1),(4095,1),(length-2,2)]:
            cache = DynamicCache(config=model.config)
            for i in range(model.config.num_hidden_layers):
                k,v = [torch.from_numpy(initial[f'past_{kind}_{i}'][:,:,:position].copy()).float() for kind in ('k','v')]
                cache.update(k,v,i)
            scores_list,state_list = [],[]
            for current in range(position,position+steps):
                result = model(input_ids=torch.tensor([[boundary_tokens[current]]],dtype=torch.int64),
                               position_ids=torch.tensor([[current]],dtype=torch.int64),past_key_values=cache,use_cache=True)
                cache = result.past_key_values
                score = result.logits[0,-1].float().numpy().copy()
                assert np.isfinite(score).all()
                scores_list.append(score)
                state_list.append(slots(cache))
            name = f'long-position-{position}'
            np.savez(out_dir / (name+'.npz'),tokens=np.array(boundary_tokens[position:position+steps],dtype=np.int64),
                     logits=np.stack(scores_list),slots=np.stack(state_list))
            cases.append({'name':name,'steps':steps,'start_position':position,
                          'initial_cache':'long-prefix-fp16.npz','initial_cache_prefix':position,
                          'reference_dtype':'float32','cache_mode':'real chunked FP32 prefix, cast to FP16 and shared as seed',
                          'require_all_greedy_match':True})
            print('Native high-position reference:',name,'steps:',steps,flush=True)
            del cache,result,k,v,scores_list,state_list
            gc.collect()
    write_json(out_dir / 'cases.json',cases)
    print('Independent FP32 reference complete, including positions 1023, 4095, 8190 and 8191.',flush=True)



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


def precision_check(scores,state,expected_scores,expected_state,thresholds):
    import numpy as np
    metric = error_metrics(scores,expected_scores)
    cache_metric = error_metrics(state,expected_state)
    delta = np.abs(state.astype(np.float64)-expected_state.astype(np.float64))
    budget = thresholds['cache_atol']+thresholds['cache_rtol']*np.abs(expected_state.astype(np.float64))
    cache_metric['max_scaled_error_ratio'] = float(np.max(delta/budget))
    cache_metric['max_layer_rmse'] = float(np.sqrt(np.mean(delta*delta,axis=(-2,-1))).max())
    failures = []
    if metric['max_abs'] > thresholds['logits_max_abs'] or metric['rmse'] > thresholds['logits_rmse'] or metric['cosine'] < thresholds['logits_cosine_min']:
        failures.append('logits')
    if cache_metric['max_scaled_error_ratio'] > 1 or cache_metric['rmse'] > thresholds['cache_global_rmse'] or cache_metric['max_layer_rmse'] > thresholds['cache_layer_rmse']:
        failures.append('cache_slot')
    return metric,cache_metric,failures


def validate(args, request):
    import gc
    import numpy as np
    import onnx
    import onnxruntime as ort
    local,shared = Path(args.local),Path(args.shared)
    graph_path = local / 'onnx/qwen3_06b_ctx8192.onnx'
    onnx.checker.check_model(str(graph_path))
    graph = onnx.load(str(graph_path),load_external_data=False)
    assert [(item.domain,item.version) for item in graph.opset_import] == [('',request['opset'])]
    cfg,length = request['config'],request['context']
    layers,heads,dim = cfg['num_hidden_layers'],cfg['num_key_value_heads'],cfg['head_dim']
    names = [f'past_{kind}_{i}' for i in range(layers) for kind in ('k','v')]
    initializers = {x.name for x in graph.graph.initializer}
    shapes,interface = [],{'inputs':[],'outputs':[]}
    for item in graph.graph.input:
        if item.name in initializers:
            continue
        shape = [d.dim_value for d in item.type.tensor_type.shape.dim]
        expected_shape = [1,1] if item.name == 'input_ids' else [1] if item.name == 'position' else [1,heads,length,dim]
        dtype = onnx.TensorProto.INT64 if item.name in ('input_ids','position') else onnx.TensorProto.FLOAT16
        assert shape == expected_shape and item.type.tensor_type.elem_type == dtype
        shapes.append(item.name+':'+','.join(map(str,shape)))
        interface['inputs'].append({'name':item.name,'shape':shape,'dtype':onnx.TensorProto.DataType.Name(dtype)})
    for item in graph.graph.output:
        interface['outputs'].append({'name':item.name,'shape':[d.dim_value for d in item.type.tensor_type.shape.dim],
                                     'dtype':onnx.TensorProto.DataType.Name(item.type.tensor_type.elem_type)})
    assert {x['name'] for x in interface['inputs']} == set(names)|{'input_ids','position'}
    write_json(shared / 'interface.json',interface)
    (shared / 'input-shape.txt').write_text(';'.join(shapes)+'\n',encoding='utf-8')
    del graph
    gc.collect()
    options = ort.SessionOptions()
    options.intra_op_num_threads,options.inter_op_num_threads = 8,1
    session = ort.InferenceSession(str(graph_path),sess_options=options,providers=['CPUExecutionProvider'])
    output_names = ['logits']+[name.replace('past_','present_') for name in names]
    assert {item.name for item in session.get_outputs()} == set(output_names)
    thresholds = request['validation']
    report = {'status':'running','checker':'passed','reference_dtype':'float32','thresholds':thresholds,
              'cases':[],'failures':[],
              'scope':'Two short multi-token chats and four high-position decode steps using real seeded prefixes; '
                      'does not establish end-to-end ONNX generation across 8192 tokens or NPU accuracy.'}
    write_json(shared / 'validation.json',report)
    for case in load(local / 'reference/cases.json'):
        with np.load(local / 'reference' / (case['name']+'.npz'),allow_pickle=False) as saved:
            arrays = {key:saved[key].copy() for key in saved.files}
        if case.get('initial_cache'):
            with np.load(local / 'reference' / case['initial_cache'],allow_pickle=False) as seed:
                cache = {name:seed[name] for name in names}
            prefix = case['initial_cache_prefix']
            for value in cache.values():
                value[:,:,prefix:] = 0
        else:
            cache = {name:np.zeros((1,heads,length,dim),dtype=np.float16) for name in names}
        predicted,metrics,cache_metrics = [],[],[]
        for step,token in enumerate(arrays['tokens']):
            position = case.get('start_position',0)+step
            assert 0 <= position < length
            values = session.run(output_names,{'input_ids':np.array([[token]],dtype=np.int64),
                                               'position':np.array([position],dtype=np.int64),**cache})
            assert values[0].shape == (1,1,cfg['vocab_size']) and values[0].dtype == np.float16
            scores = values[0][0,-1].astype(np.float32)
            state = []
            for i in range(layers):
                pair = []
                for j in range(2):
                    name = names[2*i+j]
                    updated = values[1+2*i+j]
                    assert updated.shape == (1,heads,length,dim) and updated.dtype == np.float16 and np.isfinite(updated).all()
                    assert np.array_equal(updated[:,:,:position],cache[name][:,:,:position]),f'Previous cache overwritten: {name}'
                    assert np.array_equal(updated[:,:,position+1:],cache[name][:,:,position+1:]),f'Future cache overwritten: {name}'
                    pair.append(updated[0,:,position].astype(np.float32))
                    cache[name] = updated
                state.append(pair)
            metric,cache_metric,failures = precision_check(scores,np.asarray(state),arrays['logits'][step],arrays['slots'][step],thresholds)
            metrics.append(metric)
            cache_metrics.append(cache_metric)
            report['failures'].extend({'case':case['name'],'position':position,'reason':failure,
                                       'logits':metric,'cache_slot':cache_metric} for failure in failures)
            predicted.append(int(scores.argmax()))
            print(f'ONNX {case["name"]}: {step+1}/{len(arrays["tokens"])}, position {position}, RMSE={metric["rmse"]:.8f}',flush=True)
        offset = case.get('prompt_tokens',1)-1
        expected = arrays['logits'][offset:].argmax(axis=-1).tolist()
        generated = predicted[offset:]
        greedy_match = generated == expected
        if 'generated_tokens' in case:
            greedy_match = greedy_match and generated == case['generated_tokens']
        if not greedy_match:
            report['failures'].append({'case':case['name'],'reason':'greedy mismatch','actual':generated,'expected':expected})
        result = {'name':case['name'],'steps':len(predicted),'start_position':case.get('start_position',0),
                  'greedy_match':greedy_match,'greedy_tokens':generated,'reference_tokens':expected,
                  'past_and_future_cache_preserved':True,
                  'logits_max_abs':max(x['max_abs'] for x in metrics),'logits_max_rmse':max(x['rmse'] for x in metrics),
                  'logits_min_cosine':min(x['cosine'] for x in metrics),
                  'cache_max_scaled_error_ratio':max(x['max_scaled_error_ratio'] for x in cache_metrics),
                  'cache_max_rmse':max(x['rmse'] for x in cache_metrics),
                  'cache_max_layer_rmse':max(x['max_layer_rmse'] for x in cache_metrics),
                  'logit_metrics':metrics,'cache_metrics':cache_metrics}
        report['cases'].append(result)
        write_json(shared / 'validation.json',report)
        print('Case result:',json.dumps({key:value for key,value in result.items() if key not in ('logit_metrics','cache_metrics')}),flush=True)
        del cache,values,arrays,state
        gc.collect()
    assert {c['start_position'] for c in report['cases']} == {0,1023,4095,8190}
    report['status'] = 'passed' if not report['failures'] else 'failed'
    write_json(shared / 'validation.json',report)
    assert report['status'] == 'passed','Numerical validation failed; ATC remains blocked. See validation.json.'
    print('8192-context checker, multi-token FP32 alignment and high-position checks passed.',flush=True)



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
        assert shutil.disk_usage(local).free > 20 * 1024**3, 'Need at least 20 GiB free on the Linux build drive'
        meminfo = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        available_ram = int(meminfo['MemAvailable'].split()[0]) * 1024
        assert available_ram > 20 * 1024**3, 'This 8192-context build requires at least 20 GiB currently available WSL RAM'
        assert request['context'] == 8192 and request['config']['max_position_embeddings'] >= 8192
        assert shutil.disk_usage(staging).free > 12 * 1024**3, 'Need at least 12 GiB free on the Windows shared output drive'
        manifest['host'] = {'machine': os.uname().machine, 'kernel': os.uname().release,
                            'available_ram_at_start': available_ram, 'free_disk_at_start': shutil.disk_usage(local).free}
        manifest['versions'] = {name: importlib.metadata.version(name) for name in (
            'torch', 'transformers', 'onnx', 'onnxruntime', 'numpy', 'safetensors', 'huggingface_hub', 'accelerate')}
        assert manifest['versions'] == environment['versions'], 'Dependency versions changed since environment verification'
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
        assert sha(original_path) == request['upstream_exporter_sha256'], 'Pinned exporter changed'
        original = original_path.read_text(encoding='utf-8')
        assert original.count('trust_remote_code=True') == 1
        adapted = original.replace('trust_remote_code=True', 'trust_remote_code=False, local_files_only=True, attn_implementation="eager"')
        adapted = adapted.replace('import torch\n', 'import torch\ntorch.set_num_threads(8)\n', 1)
        session_line = 'sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])'
        assert adapted.count(session_line) == 1
        adapted = adapted.replace(session_line, 'options = ort.SessionOptions()\n        options.intra_op_num_threads = 8\n        options.inter_op_num_threads = 1\n        sess = ort.InferenceSession(args.output, sess_options=options, providers=["CPUExecutionProvider"])')
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
        write_json(shared / 'validation-policy.json', request['validation'])
        write_json(shared / 'memory-estimate.json', request['memory_estimate'])
        common = ['--request', str(local / 'request.json'), '--local', str(local), '--shared', str(shared)]
        # The native reference is a fresh process and never imports the exporter.
        command('native_reference', [sys.executable, '-u', driver, 'reference', *common], cwd=project)
        command('export', [sys.executable, '-u', exporter, '--model-path', snapshot, '--max-len', str(request['context']),
                           '--output', local / 'onnx/qwen3_06b_ctx8192.onnx'], cwd=project)
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
        atc_args = ['--model=/model/qwen3_06b_ctx8192.onnx', '--framework=5', '--output=/output/qwen3_06b_ctx8192',
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
        om = local / 'om/qwen3_06b_ctx8192.om'
        assert om.is_file() and om.stat().st_size > 0
        assert 'ATC run success' in (shared / 'logs/atc.log').read_text(encoding='utf-8')
        phase('deliver')
        shutil.copy2(om, shared / 'deliverables' / om.name)
        tokenizer = shared / 'deliverables/tokenizer'
        tokenizer.mkdir()
        for record in request['files']:
            if not record['name'].endswith(('.safetensors', '.bin', '.pt', '.pth')):
                shutil.copy2(snapshot / record['name'], tokenizer / record['name'])
        for record_name in ('interface.json', 'input-shape.txt', 'validation.json', 'validation-policy.json', 'memory-estimate.json'):
            shutil.copy2(shared / record_name, shared / 'deliverables' / record_name)
        shutil.copy2(shared / 'logs/atc.log', shared / 'deliverables/atc.log')
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
        report = f"# Qwen3 本地转换记录\n\n状态：{manifest['status']}\n\n目标：Ascend310B4，静态总上下文 8192，CPU FP16 导出，ONNX opset 15。\n\n原始模型：{request['model_relative_path']}；上游 revision 未知，以输入文件 SHA256 固定本地版本。\n\nLinux 工作目录：{local}\n\n当前/最后阶段：{manifest['stage']}\n\n错误：{manifest.get('error', '无')}\n\n各阶段退出码、独立原生 FP32 参考与预设数值阈值、缓存验证、实际 ATC 参数及文件哈希见 build-manifest.json 与 validation.json。\n\n导出器修改只用于本次构建，关闭远程代码及联网加载并固定线程数，补丁见 source/export.patch。\n\nONNX 与全部外部数据保留在 onnx/。成功时 OM 和匹配的 tokenizer/config 位于 deliverables/。\n\n8192 是输入与生成共用的总容量。参考前缀分块计算，长位置检查复用真实前缀种子，不是完整8192步ONNX连续生成验证。尚未执行开发板推理，board_runtime_verified=false；板端内存、准确率和性能仍待实测。\n"
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
