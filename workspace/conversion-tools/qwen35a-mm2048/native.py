"""Independent native FP32 reference. Never imports the export adapter."""
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
import torch
from contract import image_patches, layer_groups, positions, prompt_tokens


def arrays(cache, config, length, floating='float32', slots_only=False):
    linear, attention = layer_groups(config)
    result = {}
    for ordinal, index in enumerate(linear):
        layer = cache.layers[index]
        result[f's_past_{ordinal}'] = layer.recurrent_states.detach().float().numpy().copy()
        result[f'c_past_{ordinal}'] = layer.conv_states.detach().float().numpy().astype(floating)
        assert result[f'c_past_{ordinal}'].shape[-1] == 4
    for ordinal, index in enumerate(attention):
        layer = cache.layers[index]
        for kind, tensor in (('k', layer.keys), ('v', layer.values)):
            tensor = tensor.detach().float().numpy()
            if slots_only:
                value = tensor[:, :, -1:].copy()
            else:
                value = np.zeros((1, tensor.shape[1], length, tensor.shape[3]), dtype=floating)
                value[:, :, :tensor.shape[2]] = tensor
            result[f'{kind}_past_{ordinal}'] = value
    assert all(np.isfinite(value).all() for value in result.values())
    return result


def native_positions(model, ids, grid=None):
    ids_tensor = torch.tensor([ids], dtype=torch.int64)
    mask = (ids_tensor == model.config.image_token_id).long()
    actual, _ = model.model.get_rope_index(input_ids=ids_tensor,
                                          image_grid_thw=None if grid is None else torch.tensor([grid]),
                                          mm_token_type_ids=mask)
    expected = positions(ids, model.config.to_dict(), grid or (1, 48, 48))
    assert np.array_equal(actual.numpy(), expected), 'Host image/text position construction differs from native MRoPE'
    return expected


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def reference(args):
    assert 'adapter' not in sys.modules
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    root, package = Path(args.local), Path(args.package)
    request = json.loads((root / 'request.json').read_text())
    cfg = request['config']
    out = root / 'reference'
    out.mkdir()
    model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(root / 'model'), dtype=torch.float32, device_map='cpu', local_files_only=True,
        trust_remote_code=False, attn_implementation='eager', low_cpu_mem_usage=True, output_loading_info=True)
    assert not any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')), loading
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(root / 'model'), local_files_only=True, trust_remote_code=False)
    language = model.model.language_model
    assert model.config.text_config._attn_implementation == 'eager'
    assert model.config.vision_config._attn_implementation == 'eager'
    assert sum(p.numel() for p in model.parameters()) == 2213241664
    cases, image_features, pixels = [], {}, {}
    with torch.inference_mode():
        for fixture in ('shapes', 'checker'):
            rgb = np.load(package / 'assets' / (fixture + '.npy'), allow_pickle=False)
            patches = image_patches(rgb)
            pixels[fixture] = torch.from_numpy(patches)
            output = model.model.visual(pixels[fixture], grid_thw=torch.tensor([[1, 48, 48]])).pooler_output
            assert output.shape == (576, 2048) and torch.isfinite(output).all()
            image_features[fixture] = output
            np.save(out / (fixture + '-vision.npy'), output.numpy())
            print('Native FP32 vision reference:', fixture, flush=True)

        def point(case_folder, cache, logits, position):
            path = case_folder / f'point-{position}.npz'
            scores = logits[0, -1].float().numpy().copy()
            np.savez(path, logits=scores, **arrays(cache, cfg, 2048, slots_only=True))
            return {'position': position, 'file': str(path.relative_to(out))}

        def standard_case(name, prompt, fixture=None):
            case_folder = out / name
            case_folder.mkdir()
            ids = prompt_tokens(tokenizer, prompt, image=fixture is not None)
            count = len(ids)
            assert count + 8 <= 2048
            rope = native_positions(model, ids, [1, 48, 48] if fixture else None)
            embeddings = language.embed_tokens(torch.tensor([ids]))
            if fixture:
                mask = torch.tensor(ids) == cfg['image_token_id']
                assert int(mask.sum()) == 576
                embeddings[0, mask] = image_features[fixture]
            cache, checkpoints = None, []
            stops = sorted({1, min(4, count), count} | set(range(64, count, 64)))
            start = 0
            for stop in stops:
                p4 = torch.cat((torch.arange(start, stop).reshape(1, 1, -1),
                                torch.from_numpy(rope[:, :, start:stop])), dim=0)
                result = language(inputs_embeds=embeddings[:, start:stop], position_ids=p4,
                                  past_key_values=cache, use_cache=True)
                cache = result.past_key_values
                logits = model.lm_head(result.last_hidden_state[:, -1:])
                checkpoints.append(point(case_folder, cache, logits, stop - 1))
                print(f'Native {name} prefill: {stop}/{count}', flush=True)
                start = stop
            if fixture:
                # Independent full multimodal forward proves image insertion and chunked native hand-off.
                full = model(input_ids=torch.tensor([ids]), pixel_values=pixels[fixture],
                             image_grid_thw=torch.tensor([[1, 48, 48]]),
                             mm_token_type_ids=(torch.tensor([ids]) == cfg['image_token_id']).long(),
                             position_ids=torch.cat((torch.arange(count).reshape(1, 1, -1), torch.from_numpy(rope)), dim=0),
                             use_cache=False, logits_to_keep=1)
                delta = (full.logits - logits).double()
                equivalence = {'max_abs': float(delta.abs().max()), 'rmse': float(delta.square().mean().sqrt()),
                               'greedy_equal': int(full.logits.argmax()) == int(logits.argmax())}
                assert equivalence['max_abs'] < .01 and equivalence['rmse'] < .002 and equivalence['greedy_equal'], equivalence
                save_json(case_folder / 'native-multimodal-equivalence.json', equivalence)
                del full
            generated = [int(logits[0, -1].argmax())]
            next_rope = int(rope.max()) + 1
            for step in range(7):
                token = generated[-1]
                ids.append(token)
                pos = count + step
                r = np.full((3, 1, 1), next_rope + step, dtype=np.int64)
                rope = np.concatenate((rope, r), axis=-1)
                p4 = torch.cat((torch.tensor([[[pos]]]), torch.from_numpy(r)), dim=0)
                result = language(input_ids=torch.tensor([[token]]), position_ids=p4,
                                  past_key_values=cache, use_cache=True)
                cache = result.past_key_values
                logits = model.lm_head(result.last_hidden_state)
                checkpoints.append(point(case_folder, cache, logits, pos))
                generated.append(int(logits[0, -1].argmax()))
            np.savez(case_folder / 'inputs.npz', ids=np.array(ids, dtype=np.int64), rope=rope)
            cases.append({'name': name, 'inputs': name + '/inputs.npz', 'start_position': 0,
                          'prompt_tokens': count, 'generated_tokens': generated,
                          'generated_text': tokenizer.decode(generated, skip_special_tokens=True),
                          'fixture': fixture, 'checkpoints': checkpoints})
            print('Native case complete:', name, 'greedy:', generated, flush=True)

        standard_case('text-zh', '你好，请用一句话介绍你自己。')
        standard_case('text-en', 'Write a short greeting.')
        standard_case('image-shapes', 'Describe the colors and shapes in this picture briefly.', 'shapes')

        pattern = tokenizer.encode('This is a context boundary validation sequence. ', add_special_tokens=False)
        tokens = (pattern * ((2048 + len(pattern) - 1) // len(pattern)))[:2048]
        cache, start = None, 0
        linear, attention = layer_groups(cfg)
        for boundary, steps in ((511, 1), (1023, 1), (2046, 2)):
            while start < boundary:
                stop = min(start + 64, boundary)
                result = language(input_ids=torch.tensor([tokens[start:stop]]),
                                  position_ids=torch.arange(start, stop).reshape(1, -1),
                                  past_key_values=cache, use_cache=True)
                cache, start = result.past_key_values, stop
                if stop % 256 == 0 or stop == boundary:
                    print('Native real prefix:', stop, '/2046', flush=True)
            name = f'boundary-{boundary}'
            case_folder = out / name
            case_folder.mkdir()
            # Each recurrent prefix gets its OWN state: a later state cannot be truncated to an earlier position.
            seed = arrays(cache, cfg, 2048, floating='float16')
            np.savez(case_folder / 'seed.npz', **seed)
            probe = copy.deepcopy(cache)
            for index in linear:
                probe.layers[index].conv_states = probe.layers[index].conv_states.half().float()
            for index in attention:
                probe.layers[index].keys = probe.layers[index].keys.half().float()
                probe.layers[index].values = probe.layers[index].values.half().float()
            checkpoints = []
            for pos in range(boundary, boundary + steps):
                result = language(input_ids=torch.tensor([[tokens[pos]]]), position_ids=torch.tensor([[pos]]),
                                  past_key_values=probe, use_cache=True)
                probe = result.past_key_values
                logits = model.lm_head(result.last_hidden_state)
                checkpoints.append(point(case_folder, probe, logits, pos))
            ids = np.array(tokens[boundary:boundary + steps], dtype=np.int64)
            rope = np.broadcast_to(np.arange(boundary, boundary + steps)[None, None, :], (3, 1, steps)).copy()
            np.savez(case_folder / 'inputs.npz', ids=ids, rope=rope)
            cases.append({'name': name, 'inputs': name + '/inputs.npz', 'start_position': boundary,
                          'seed': name + '/seed.npz', 'fixture': None, 'checkpoints': checkpoints})
            del probe
    save_json(out / 'cases.json', cases)
    save_json(out / 'reference-status.json', {'status': 'passed', 'reference_dtype': 'float32',
                                             'unpatched_model': True, 'image_grid_thw': [1, 48, 48]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', required=True)
    parser.add_argument('--package', required=True)
    torch.set_num_threads(8)
    reference(parser.parse_args())
