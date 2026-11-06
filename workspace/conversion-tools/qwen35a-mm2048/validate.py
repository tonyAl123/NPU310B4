"""CPU ONNX checker, independent FP32 comparisons and two-subgraph image/text replay."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from contract import image_patches, output_names, state_spec, zero_state


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(4 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def metric(actual, expected):
    a, b = np.asarray(actual, dtype=np.float64).ravel(), np.asarray(expected, dtype=np.float64).ravel()
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    delta = a - b
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return {'max_abs': float(np.abs(delta).max()), 'rmse': float(np.sqrt(np.mean(delta * delta))),
            'cosine': float(np.dot(a, b) / denominator) if denominator else float(np.array_equal(a, b))}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def inspect_graph(path):
    onnx.checker.check_model(str(path))
    graph = onnx.load(str(path), load_external_data=False)
    assert [(x.domain, x.version) for x in graph.opset_import] == [('', 15)]
    initializers = {x.name for x in graph.graph.initializer}
    interface = {'inputs': [], 'outputs': [], 'external_data': {}}
    for label, values in (('inputs', graph.graph.input), ('outputs', graph.graph.output)):
        for value in values:
            if label == 'inputs' and value.name in initializers:
                continue
            dims = [d.dim_value for d in value.type.tensor_type.shape.dim]
            interface[label].append({'name': value.name, 'shape': dims,
                                     'dtype': onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type)})
    for value in graph.graph.initializer:
        if value.data_location == onnx.TensorProto.EXTERNAL:
            fields = {item.key: item.value for item in value.external_data}
            relative = Path(fields['location'])
            assert not relative.is_absolute() and '..' not in relative.parts
            data = path.parent / relative
            assert data.resolve().is_relative_to(path.parent.resolve()) and data.is_file()
            offset, count = int(fields.get('offset', 0)), int(fields['length'])
            assert 0 <= offset < data.stat().st_size and 0 < count <= data.stat().st_size - offset
            if str(relative) not in interface['external_data']:
                interface['external_data'][str(relative)] = {'bytes': data.stat().st_size, 'sha256': sha(data)}
    assert interface['external_data']
    return interface


def run(args):
    root, shared, package = Path(args.local), Path(args.shared), Path(args.package)
    request = json.loads((root / 'request.json').read_text())
    config, policy = request['config'], request['validation']
    report = {'status': 'running', 'scope': policy['scope'], 'policy': policy,
              'vision': [], 'cases': [], 'failures': [], 'board_runtime_verified': False,
              'purpose': 'diagnostic_native_vision' if args.reference_vision else 'full_pipeline_gate',
              'eligible_for_atc': not args.reference_vision, 'vision_input_dtype': args.vision_input_dtype}
    interfaces = {}
    for graph in ('vision', 'decoder'):
        info = inspect_graph(root / 'onnx' / graph / 'model.onnx')
        interfaces[graph] = info
        write(shared / f'{graph}-interface.json', info)
        shape = ';'.join(item['name'] + ':' + ','.join(map(str, item['shape'])) for item in info['inputs'])
        (shared / f'{graph}-input-shape.txt').write_text(shape + '\n')
    expected_inputs = [('input_ids', [1, 1], 'INT64'), ('position', [1], 'INT64'),
                       ('rope_position', [3, 1, 1], 'INT64'), ('image_embed', [1, 1, 2048], 'FLOAT16'),
                       ('is_image', [1], 'INT64')]
    expected_inputs += [(n, s, 'FLOAT' if d == 'float32' else 'FLOAT16') for n, s, d in state_spec(config, 2048)]
    assert [(x['name'], x['shape'], x['dtype']) for x in interfaces['decoder']['inputs']] == expected_inputs
    vision_dtype = 'FLOAT' if args.vision_input_dtype == 'float32' else 'FLOAT16'
    assert interfaces['vision']['inputs'] == [{'name': 'pixel_values', 'shape': [2304, 1536], 'dtype': vision_dtype}]
    assert interfaces['vision']['outputs'] == [{'name': 'image_features', 'shape': [576, 2048], 'dtype': 'FLOAT16'}]
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 8, 1
    encoder = None if args.reference_vision else ort.InferenceSession(
        str(root / 'onnx/vision/model.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
    features = {}
    for fixture in ('shapes', 'checker'):
        rgb = np.load(package / 'assets' / (fixture + '.npy'), allow_pickle=False)
        expected = np.load(root / 'reference' / (fixture + '-vision.npy'), allow_pickle=False)
        actual = expected.astype(np.float16) if args.reference_vision else encoder.run(
            ['image_features'], {'pixel_values': image_patches(rgb).astype(args.vision_input_dtype)})[0]
        assert actual.shape == (576, 2048) and actual.dtype == np.float16
        result = metric(actual, expected)
        passed = result['max_abs'] <= policy['vision_max_abs'] and result['rmse'] <= policy['vision_rmse'] and result['cosine'] >= policy['vision_cosine_min']
        report['vision'].append({'fixture': fixture, 'passed': passed, **result})
        if not passed:
            report['failures'].append({'fixture': fixture, 'reason': 'vision features'})
        features[fixture] = actual
        print('Vision FP32 comparison:', fixture, result, flush=True)
    del encoder
    session = ort.InferenceSession(str(root / 'onnx/decoder/model.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
    out_names = output_names(config)
    assert [v.name for v in session.get_outputs()] == out_names
    write(shared / 'validation.json', report)
    reference = root / 'reference'
    for case in json.loads((reference / 'cases.json').read_text()):
        case_inputs = np.load(reference / case['inputs'], allow_pickle=False)
        ids, rope = case_inputs['ids'], case_inputs['rope']
        state = zero_state(config, 2048)
        if case.get('seed'):
            with np.load(reference / case['seed'], allow_pickle=False) as saved:
                state = {name: saved[name].copy() for name in state}
        checkpoints = {point['position']: point for point in case['checkpoints']}
        image_index, generated, points, case_pass = 0, [], [], True
        for step, token in enumerate(ids):
            pos = case['start_position'] + step
            assert 0 <= pos < 2048
            image = bool(case.get('fixture')) and step < case['prompt_tokens'] and int(token) == config['image_token_id']
            image_embed = np.zeros((1, 1, 2048), dtype=np.float16)
            if image:
                assert case['fixture']
                image_embed[:] = features[case['fixture']][image_index]
                image_index += 1
            values = session.run(out_names, {'input_ids': np.array([[token]], np.int64),
                'position': np.array([pos], np.int64), 'rope_position': rope[:, :, step:step+1],
                'image_embed': image_embed, 'is_image': np.array([int(image)], np.int64), **state})
            assert values[0].shape == (1, 1, 248320) and values[0].dtype == np.float16 and np.isfinite(values[0]).all()
            updated_state = {}
            for (name, shape, dtype), value in zip(state_spec(config, 2048), values[1:]):
                assert value.shape == tuple(shape) and value.dtype == np.dtype(dtype) and np.isfinite(value).all(), name
                if name.startswith(('k_', 'v_')):
                    assert np.array_equal(value[:, :, :pos], state[name][:, :, :pos]), name + ' past overwritten'
                    assert np.array_equal(value[:, :, pos+1:], state[name][:, :, pos+1:]), name + ' future overwritten'
                updated_state[name] = value
            state = updated_state
            if 'prompt_tokens' in case and pos >= case['prompt_tokens'] - 1:
                generated.append(int(values[0].argmax()))
            if pos in checkpoints:
                with np.load(reference / checkpoints[pos]['file'], allow_pickle=False) as expected:
                    scores = metric(values[0][0, -1], expected['logits'])
                    passed = scores['max_abs'] <= policy['logits_max_abs'] and scores['rmse'] <= policy['logits_rmse'] and scores['cosine'] >= policy['logits_cosine_min']
                    state_worst = {'max_scaled_error': 0., 'max_rmse': 0.}
                    for name, value in state.items():
                        actual = value[:, :, pos:pos+1] if name.startswith(('k_', 'v_')) else value
                        target = expected[name]
                        difference = np.abs(actual.astype(np.float64) - target.astype(np.float64))
                        recurrent = name.startswith('s_')
                        atol = policy['recurrent_atol'] if recurrent else policy['cache_atol']
                        rtol = policy['recurrent_rtol'] if recurrent else policy['cache_rtol']
                        ratio = float(np.max(difference / (atol + rtol * np.abs(target.astype(np.float64)))))
                        rmse = float(np.sqrt(np.mean(difference * difference)))
                        limit = policy['recurrent_rmse'] if recurrent else policy['cache_rmse']
                        if ratio > 1 or rmse > limit:
                            passed = False
                            report['failures'].append({'case': case['name'], 'position': pos, 'state': name,
                                                        'scaled_error': ratio, 'rmse': rmse})
                        state_worst['max_scaled_error'] = max(state_worst['max_scaled_error'], ratio)
                        state_worst['max_rmse'] = max(state_worst['max_rmse'], rmse)
                    if 'prompt_tokens' not in case:
                        passed = passed and int(values[0].argmax()) == int(expected['logits'].argmax())
                    if not passed:
                        report['failures'].append({'case': case['name'], 'position': pos, 'reason': 'checkpoint comparison', 'logits': scores})
                    case_pass = case_pass and passed
                    points.append({'position': pos, 'passed': passed, 'logits': scores, 'state': state_worst})
            if step % 64 == 0 or step == len(ids)-1:
                print(f'ONNX {case["name"]}: {step+1}/{len(ids)}, position {pos}', flush=True)
        if 'generated_tokens' in case and generated != case['generated_tokens']:
            case_pass = False
            report['failures'].append({'case': case['name'], 'reason': 'generated greedy mismatch',
                                       'expected': case['generated_tokens'], 'actual': generated})
        assert image_index == (576 if case.get('fixture') else 0)
        report['cases'].append({'name': case['name'], 'steps': len(ids), 'passed': case_pass,
                                'generated_tokens': generated, 'checkpoints': points,
                                'prior_and_future_kv_preserved': True})
        write(shared / 'validation.json', report)
    report['status'] = 'passed' if not report['failures'] else 'failed'
    write(shared / 'validation.json', report)
    assert report['status'] == 'passed', 'Numerical gate failed; ATC remains blocked. See validation.json.'
    compact = shared / 'reference-vectors'
    compact.mkdir()
    write(compact / 'cases.json', report['cases'])
    for case in json.loads((reference / 'cases.json').read_text()):
        if case.get('start_position', 0) == 0:
            import shutil
            shutil.copy2(reference / case['inputs'], compact / (case['name'] + '-inputs.npz'))
    print('Both subgraphs, full image/text prefill/decode and boundary numerical checks passed.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', required=True)
    parser.add_argument('--shared', required=True)
    parser.add_argument('--package', required=True)
    parser.add_argument('--reference-vision', action='store_true',
                        help='Diagnostic only: replay decoder with rounded native FP32 image features; never an ATC gate')
    parser.add_argument('--vision-input-dtype', choices=['float16', 'float32'], default='float16')
    run(parser.parse_args())
