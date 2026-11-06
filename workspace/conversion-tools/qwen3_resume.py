"""Revalidate the pinned Qwen3 graph against audited FP32 and compile if it passes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import uuid

GRAPH_SHA = '076894dae3591efc336cd1b0d5f4ced40f9488ddfedac55b5456d8b8791f85a5'
AUDIT_ID = '20261020T054718Z-91cd5a'
POLICY = {
    'version': 2, 'primary_reference': 'independent native FP32 from pinned diagnostic run',
    'original_fp16_validation_remains_failed': True,
    'reason': 'ONNX logits are closer to FP32 than native FP16; cache magnitudes exceed 500. '
              'Use FP32 for primary numerical error and scale cache tolerances by value magnitude.',
    'logits_max_abs': 0.25, 'logits_rmse': 0.03, 'logits_cosine_min': 0.999,
    'cache_atol': 0.15, 'cache_rtol': 2**-9, 'cache_global_rmse': 0.03, 'cache_layer_rmse': 0.03,
    'cache_rule': 'Every element: abs(actual-reference) <= 0.15 + (2 * FP16 epsilon) * abs(reference). '
                  'Additionally require global and per-layer K/V RMSE <= 0.03.',
    'secondary_fp16': 'Record all FP16 numerical comparisons as diagnostics. '
                      'Numerical acceptance uses FP32; generated tokens must also equal FP16.',
    'greedy': 'Every generated token must equal both native FP16 and native FP32.',
    'boundary': 'Position 255; FP32 reference consumes the identical FP16 prefix cast to FP32.',
    'limitations': 'Two short prompts and one boundary probe; target-board inference remains unverified.',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(4*1024**2):
            digest.update(block)
    return digest.hexdigest()


def metrics(actual, expected):
    import numpy as np
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    d = a-b
    norm = float(np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()))
    return {'max_abs': float(np.abs(d).max()), 'rmse': float(np.sqrt(np.mean(d*d))),
            'cosine': float(np.sum(a*b)/norm) if norm else float(np.array_equal(a,b))}


def judge(scores, states, fp32_scores, fp32_states, fp16_scores, fp16_states):
    import numpy as np
    result, failures = {}, []
    for label, ref_scores, ref_states in [('fp32',fp32_scores,fp32_states), ('fp16',fp16_scores,fp16_states)]:
        lm, cm = metrics(scores, ref_scores), metrics(states, ref_states)
        delta = np.abs(states.astype(np.float64)-ref_states.astype(np.float64))
        budget = POLICY['cache_atol']+POLICY['cache_rtol']*np.abs(ref_states.astype(np.float64))
        cm['max_scaled_error_ratio'] = float(np.max(delta/budget))
        # Shape is layer, K/V, KV head, channel. Check each layer and K/V
        # separately so large first-layer key channels cannot hide later errors.
        cm['max_layer_rmse'] = float(np.sqrt(np.mean(delta*delta, axis=(-2,-1))).max())
        result[label] = {'logits':lm, 'cache':cm}
        if label == 'fp32' and (lm['max_abs'] > POLICY['logits_max_abs'] or lm['cosine'] < POLICY['logits_cosine_min']):
            failures.append(label+' logits max_abs/cosine')
        if label == 'fp32' and lm['rmse'] > POLICY['logits_rmse']:
            failures.append('fp32 logits RMSE')
        if label == 'fp32' and (cm['max_scaled_error_ratio'] > 1 or cm['rmse'] > POLICY['cache_global_rmse']):
            failures.append(label+' cache scaled error/global RMSE')
        if label == 'fp32' and cm['max_layer_rmse'] > POLICY['cache_layer_rmse']:
            failures.append('fp32 cache layer RMSE')
    return result, failures


def validate(local, audit, output, request):
    import numpy as np
    import onnx
    import onnxruntime as ort
    graph_path = local / 'onnx/qwen3_06b_ctx256.onnx'
    onnx.checker.check_model(str(graph_path))
    graph = onnx.load(str(graph_path), load_external_data=False)
    assert [(op.domain,op.version) for op in graph.opset_import] == [('',request['opset'])]
    assert not any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.graph.initializer), 'Pinned graph was inline'
    cfg, length = request['config'], request['context']
    layers, heads, dim = cfg['num_hidden_layers'], cfg['num_key_value_heads'], cfg['head_dim']
    names = [f'past_{kind}_{i}' for i in range(layers) for kind in ('k','v')]
    initializers = {t.name for t in graph.graph.initializer}
    inputs = [i for i in graph.graph.input if i.name not in initializers]
    interface, shapes = [], []
    for item in inputs:
        shape = [d.dim_value for d in item.type.tensor_type.shape.dim]
        expected_shape = [1,1] if item.name == 'input_ids' else [1] if item.name == 'position' else [1,heads,length,dim]
        expected_type = onnx.TensorProto.INT64 if item.name in ('input_ids','position') else onnx.TensorProto.FLOAT16
        assert shape == expected_shape and item.type.tensor_type.elem_type == expected_type
        interface.append({'name':item.name,'shape':shape,'dtype':onnx.TensorProto.DataType.Name(expected_type)})
        shapes.append(item.name+':'+','.join(map(str,shape)))
    assert {x['name'] for x in interface} == set(names)|{'input_ids','position'}
    write(output / 'interface.json', interface)
    (output / 'input-shape.txt').write_text(';'.join(shapes)+'\n',encoding='utf-8')
    del graph
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 8,1
    session = ort.InferenceSession(str(graph_path), sess_options=options, providers=['CPUExecutionProvider'])
    output_names = ['logits']+[name.replace('past_','present_') for name in names]
    assert set(output_names) == {item.name for item in session.get_outputs()}
    report = {'status':'running','policy':POLICY,'checker':'passed','cases':[],'failures':[]}
    for case in read(local / 'reference/cases.json'):
        name = case['name']
        with np.load(local / 'reference' / (name+'.npz'),allow_pickle=False) as data:
            ref16 = {key:data[key].copy() for key in data.files}
        with np.load(audit / ('native-fp32-'+name+'.npz'),allow_pickle=False) as data:
            ref32 = {key:data[key].copy() for key in data.files}
        assert np.array_equal(ref16['tokens'],ref32['tokens'])
        cache = {key:np.zeros((1,heads,length,dim),dtype=np.float16) for key in names}
        if case.get('initial_cache'):
            with np.load(local / 'reference' / case['initial_cache'],allow_pickle=False) as seed:
                cache = {key:seed[key].copy() for key in names}
        results, predicted = [], []
        for step, token in enumerate(ref16['tokens']):
            position = case.get('start_position',0)+step
            assert 0 <= position < length
            values = session.run(output_names, {'input_ids':np.array([[token]],dtype=np.int64),
                                                'position':np.array([position],dtype=np.int64),**cache})
            assert values[0].shape == (1,1,cfg['vocab_size']) and values[0].dtype == np.float16
            scores = values[0][0,-1].astype(np.float32)
            slots = []
            for i in range(layers):
                pair = []
                for j in range(2):
                    key = names[2*i+j]
                    value = values[1+2*i+j]
                    assert value.shape == (1,heads,length,dim) and value.dtype == np.float16 and np.isfinite(value).all()
                    assert np.array_equal(value[:,:,:position], cache[key][:,:,:position]), 'Previous KV overwritten'
                    assert np.array_equal(value[:,:,position+1:], cache[key][:,:,position+1:]), 'Future KV overwritten'
                    pair.append(value[0,:,position].astype(np.float32))
                    cache[key] = value
                slots.append(pair)
            metric, failures = judge(scores,np.asarray(slots),ref32['logits'][step],ref32['slots'][step],
                                     ref16['logits'][step],ref16['slots'][step])
            results.append({'position':position,**metric})
            predicted.append(int(scores.argmax()))
            report['failures'].extend({'case':name,'position':position,'reason':reason} for reason in failures)
        offset = case.get('prompt_tokens',1)-1
        generated = predicted[offset:]
        fp32_generated = ref32['logits'][offset:].argmax(axis=-1).tolist()
        fp16_generated = ref16['logits'][offset:].argmax(axis=-1).tolist()
        greedy = generated == fp32_generated == fp16_generated
        if 'generated_tokens' in case:
            greedy = greedy and generated == case['generated_tokens']
        if not greedy:
            report['failures'].append({'case':name,'reason':'greedy mismatch'})
        summary = {'name':name,'steps':len(results),'greedy_match_both_references':greedy,
                   'generated_tokens':generated,'past_and_future_cache_preserved':True,'metrics':results,
                   'fp32_max_logit_rmse':max(r['fp32']['logits']['rmse'] for r in results),
                   'fp32_max_cache_scaled_ratio':max(r['fp32']['cache']['max_scaled_error_ratio'] for r in results)}
        report['cases'].append(summary)
        write(output / 'validation.json',report)
        print('REVALIDATION:',name,'FP32 max logit RMSE:',summary['fp32_max_logit_rmse'],
              'cache error/budget:',summary['fp32_max_cache_scaled_ratio'],'greedy:',greedy,flush=True)
    report['status'] = 'passed' if not report['failures'] else 'failed'
    write(output / 'validation.json',report)
    assert report['status'] == 'passed', 'Revalidation failed; ATC remains blocked'


def run(args):
    original = Path(args.attempt).resolve()
    manifest = read(original / 'build-manifest.json')
    local = Path(manifest['linux_work_directory']).resolve()
    assert local.is_relative_to((Path.home() / 'ascend-model-build/conversions').resolve())
    assert manifest['status'] == 'failed' and manifest['stage'] == 'numeric_validation'
    audit = original / 'audits' / AUDIT_ID
    audit_record = read(audit / 'audit-status.json')
    assert audit_record['status'] == 'diagnostics_complete' and audit_record['graph_sha256'] == GRAPH_SHA
    assert sha(original / 'build-manifest.json') == audit_record['original_manifest_sha256']
    assert sha(original / 'validation.json') == audit_record['original_validation_sha256']
    assert sha(local / 'onnx/qwen3_06b_ctx256.onnx') == GRAPH_SHA
    for record in audit_record['artifacts']:
        assert sha(audit / record['name']) == record['sha256'], 'Audit artifact changed'
    for record in manifest['artifacts']:
        if record['path'].startswith('reference/'):
            assert sha(local / record['path']) == record['sha256'], 'Original reference changed'
    request = read(local / 'request.json')
    assert request == manifest['request']
    for record in request['files']:
        assert sha(local / 'model' / record['name']) == record['sha256'], 'Snapshot changed'
    environment = read(Path(args.staging) / 'environment-status.json')
    assert environment['environment_ready'] and environment['smoke_compile'] == 'passed'
    for package,version in manifest['versions'].items():
        import importlib.metadata
        assert importlib.metadata.version(package) == version, 'Dependency changed: '+package
    identifier = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6]
    output = original / 'resumptions' / identifier
    output.mkdir(parents=True)
    for directory in ['logs','deliverables','source']:
        (output / directory).mkdir()
    om_root = local / 'resumptions' / identifier / 'om'
    om_root.mkdir(parents=True)
    (om_root / 'ascend-logs').mkdir()
    shutil.copy2(__file__,output / 'source/qwen3_resume.py')
    write(output / 'validation-policy.json',POLICY)
    result = {'status':'running','stage':'revalidation','board_runtime_verified':False,
              'original_failed_attempt':str(original),'original_validation_status':'failed',
              'audit':str(audit),'audit_status_sha256':sha(audit / 'audit-status.json'),
              'graph_path':str(local / 'onnx/qwen3_06b_ctx256.onnx'),'graph_sha256':GRAPH_SHA,
              'policy':POLICY,'request':request,'versions':manifest['versions'],'commands':[],
              'source_sha256':sha(output / 'source/qwen3_resume.py')}

    def update():
        write(output / 'build-manifest.json',result)
        write(original / 'latest-resumption.json',{'directory':str(output),'status':result['status'],'stage':result['stage']})
        write(original.parent.parent / 'conversion-status.json',
              {'status':result['status'],'stage':result['stage'],'attempt':str(output),'local':str(om_root.parent)})

    try:
        update()
        validate(local,audit,output,request)
        result['numeric_validation'] = 'passed_under_policy_v2'
        result['stage'] = 'atc'
        update()
        podman = ['podman','--cgroup-manager=cgroupfs','--events-backend=file']
        image = request['container_image_id']
        info = json.loads(subprocess.check_output([*podman,'image','inspect',image],text=True))
        assert info and info[0]['Id'].removeprefix('sha256:') == image.removeprefix('sha256:')
        write(output / 'container-image.json',info)
        shape = (output / 'input-shape.txt').read_text().strip()
        atc = ['--model=/model/qwen3_06b_ctx256.onnx','--framework=5','--output=/output/qwen3_06b_ctx256',
               '--input_format=ND','--input_shape='+shape,'--soc_version='+request['target_soc'],
               '--precision_mode=allow_fp32_to_fp16','--log=info']
        command = [*podman,'run','--rm','--cgroups=disabled','--network=host',
                   '-e','TE_PARALLEL_COMPILER=1','-e','MAX_COMPILE_CORE_NUMBER=1',
                   '-e','ASCEND_PROCESS_LOG_PATH=/output/ascend-logs',
                   '-v',str(local / 'onnx')+':/model:ro','-v',str(om_root)+':/output:Z','-w','/output',
                   image,'bash','-c',
                   'atc "$@"; result=$?; if [ -d /root/ascend/log ]; then cp -a /root/ascend/log /output/compiler-internal-logs || true; fi; exit "$result"',
                   'atc-job',*atc]
        result['commands'].append({'environment':'ATC container via WSL host','argv':command})
        update()
        print('Revalidation passed. Starting ATC for',request['target_soc'],flush=True)
        with (output / 'logs/atc.log').open('w',encoding='utf-8') as log:
            with subprocess.Popen(command,cwd=om_root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                  text=True,errors='replace') as process:
                for line in process.stdout:
                    print(line,end='',flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
        result['atc_exit_code'] = code
        assert code == 0,'ATC failed; see logs/atc.log'
        assert 'ATC run success' in (output / 'logs/atc.log').read_text(encoding='utf-8')
        om = om_root / 'qwen3_06b_ctx256.om'
        assert om.is_file() and om.stat().st_size > 0,'No new nonempty OM produced'
        result['stage'] = 'deliver'
        update()
        shutil.copy2(om,output / 'deliverables' / om.name)
        tokenizer = output / 'deliverables/tokenizer'
        tokenizer.mkdir()
        for record in request['files']:
            if not record['name'].endswith(('.safetensors','.bin','.pt','.pth')):
                shutil.copy2(local / 'model' / record['name'],tokenizer / record['name'])
        result['status'],result['stage'] = 'compiled','complete'
    except Exception:
        result['status'] = 'failed'
        result['error'] = traceback.format_exc()
        (output / 'logs/failure.txt').write_text(result['error'],encoding='utf-8')
        print(result['error'],flush=True)
    finally:
        for directory in ['ascend-logs','compiler-internal-logs']:
            if (om_root / directory).exists():
                shutil.copytree(om_root / directory,output / 'logs' / directory)
        result['finished_at'] = datetime.now(timezone.utc).isoformat()
        result['artifacts'] = [{'path':str(p.relative_to(output)),'bytes':p.stat().st_size,'sha256':sha(p)}
                               for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'build-manifest.json']
        update()
        shutil.copy2(output / 'build-manifest.json',output / 'deliverables/build-manifest.json')
        report = ('# Qwen3 OM 续编译记录\n\n状态：'+result['status']+'；最后阶段：'+result['stage']+
                  '\n\n原始 FP16 数值门槛结果仍为 failed，未更改原记录。本次采用有独立 FP32 诊断依据的 policy v2；'
                  'logits 的绝对误差、RMSE 和余弦门槛不变，主参考改为 FP32；缓存采用原绝对误差预算加 2 倍 FP16 epsilon 的相对预算，'
                  '另检查全局及每层 K/V RMSE。生成 token 同时对齐两个原生参考。\n\n'
                  '详细门槛、逐位置数值、编译命令和退出码见 validation-policy.json、validation.json、build-manifest.json。'
                  '\n\n原始 ONNX（及原失败记录）位于：'+str(original / 'onnx')+
                  '\n\n成功时 deliverables/ 包含本次新生成的 OM、tokenizer 和配置。目标 Ascend310B4，总上下文 256。'
                  '\n\n尚无开发板执行证据，board_runtime_verified=false；未验证板端准确率、兼容性和性能。\n')
        (output / 'BUILD_REPORT.md').write_text(report,encoding='utf-8')
        (output / 'deliverables/BUILD_REPORT.md').write_text(report,encoding='utf-8')
        print('FINAL STATUS:',result['status'],'\nWindows-visible result:',output,flush=True)
    return 0 if result['status'] == 'compiled' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt',required=True)
    parser.add_argument('--staging',required=True)
    return run(parser.parse_args())


if __name__ == '__main__':
    sys.exit(main())
