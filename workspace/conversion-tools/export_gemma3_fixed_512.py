"""Export the inspected Gemma3 text model with fixed full-storage KV buffers."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn
from transformers import AutoModelForCausalLM


class FixedCache:
    """Tensor-only insert; the native attention modules retain their own math."""
    def __init__(self, previous, position, length):
        self.previous = previous
        self.updated = [None] * len(previous)
        index = torch.arange(length, dtype=torch.int64, device=position.device)
        self.insert_mask = (index == position.reshape(1)).reshape(1, 1, length, 1)

    def update(self, keys, values, layer_idx, *args, **kwargs):
        pair = []
        for offset, value in enumerate((keys, values)):
            index = 2 * layer_idx + offset
            updated = torch.where(self.insert_mask, value, self.previous[index])
            self.updated[index] = updated
            pair.append(updated)
        return tuple(pair)


class GemmaFixedWrapper(nn.Module):
    def __init__(self, model, length):
        super().__init__()
        self.model = model
        self.length = length
        cfg = model.config
        assert cfg.model_type == 'gemma3_text' and not cfg.use_bidirectional_attention
        assert cfg.attn_logit_softcapping is None and cfg.final_logit_softcapping is None
        assert set(cfg.layer_types) == {'full_attention', 'sliding_attention'}
        assert all(cfg.rope_parameters[k]['rope_type'] == 'default' for k in set(cfg.layer_types))

    def forward(self, input_ids, position, *previous):
        # Native embedding includes Gemma's embedding scale. Native decoder
        # layers preserve all four RMSNorms, Q/K norms, MLP and residuals.
        hidden = self.model.model.embed_tokens(input_ids)
        position_ids = position.reshape(1, 1)
        indices = torch.arange(self.length, dtype=torch.int64, device=position.device)
        causal = indices <= position.reshape(1)
        local = causal & (indices > position.reshape(1) - self.model.config.sliding_window)
        zero = torch.zeros((self.length,), dtype=hidden.dtype, device=hidden.device)
        blocked = torch.full_like(zero, float('-inf'))
        masks = {kind: torch.where(allowed, zero, blocked).reshape(1, 1, 1, self.length)
                 for kind, allowed in [('full_attention', causal), ('sliding_attention', local)]}
        rotations = {kind: self.model.model.rotary_emb(hidden, position_ids, kind)
                     for kind in ('full_attention', 'sliding_attention')}
        cache = FixedCache(previous, position, self.length)
        for index, layer in enumerate(self.model.model.layers):
            kind = self.model.config.layer_types[index]
            hidden = layer(hidden, position_embeddings=rotations[kind], attention_mask=masks[kind],
                           position_ids=position_ids, past_key_values=cache)
        logits = self.model.lm_head(self.model.model.norm(hidden))
        return (logits, *cache.updated)


def bind_external_weights(graph, wrapper, folder, input_names, weight_basename, minimum_bytes):
    # Exporting without parameter payloads avoids constructing a >2 GiB
    # protobuf. Bind each named parameter back as an external initializer.
    state = wrapper.state_dict()
    state.update(dict(wrapper.named_buffers(remove_duplicate=False)))
    parameter_inputs = [item for item in graph.graph.input if item.name not in input_names]
    unknown = [item.name for item in parameter_inputs if item.name not in state]
    assert not unknown, 'Unrecognized exported parameter inputs: ' + repr(unknown[:20])
    weight_file = folder / weight_basename
    assert not weight_file.exists()
    bindings, shared = [], {}
    with weight_file.open('wb') as stream:
        for item in parameter_inputs:
            value = state[item.name].detach().cpu().contiguous()
            array = value.numpy()
            expected = [dimension.dim_value for dimension in item.type.tensor_type.shape.dim]
            dtype = onnx.helper.np_dtype_to_tensor_dtype(array.dtype)
            assert list(array.shape) == expected and dtype == item.type.tensor_type.elem_type
            key = (value.data_ptr(), tuple(value.shape), str(value.dtype))
            if key in shared:
                offset, size = shared[key]
            else:
                offset, size = stream.tell(), array.nbytes
                assert stream.write(memoryview(array).cast('B')) == size
                shared[key] = (offset, size)
            tensor = onnx.TensorProto()
            tensor.name, tensor.data_type = item.name, dtype
            tensor.dims.extend(array.shape)
            tensor.data_location = onnx.TensorProto.EXTERNAL
            for k, v in [('location', weight_file.name), ('offset', str(offset)), ('length', str(size))]:
                entry = tensor.external_data.add()
                entry.key, entry.value = k, v
            graph.graph.initializer.append(tensor)
            bindings.append({'name': item.name, 'shape': list(array.shape), 'offset': offset, 'bytes': size})
    assert parameter_inputs and weight_file.stat().st_size >= minimum_bytes
    retained = [item for item in graph.graph.input if item.name in input_names]
    assert {item.name for item in retained} == set(input_names)
    del graph.graph.input[:]
    graph.graph.input.extend(retained)
    (folder / 'weight-bindings.json').write_text(json.dumps(bindings, indent=2) + '\n', encoding='utf-8')


def export_external(model, length, output, minimum_bytes):
    cfg = model.config
    wrapper = GemmaFixedWrapper(model, length).eval()
    names = ['input_ids', 'position'] + [f'past_{kind}_{i}' for i in range(cfg.num_hidden_layers) for kind in ('k', 'v')]
    output_names = ['logits'] + [f'present_{kind}_{i}' for i in range(cfg.num_hidden_layers) for kind in ('k', 'v')]
    dtype = next(model.parameters()).dtype
    previous = [torch.zeros((1, cfg.num_key_value_heads, length, cfg.head_dim), dtype=dtype) for _ in range(2*cfg.num_hidden_layers)]
    output = Path(output).resolve()
    unbound = output.with_name('gemma3-unbound.onnx')
    assert not output.exists() and not unbound.exists()
    print('Exporting Gemma-native layers with explicit local/full masks and fixed KV storage.', flush=True)
    torch.onnx.export(wrapper, (torch.tensor([[2]], dtype=torch.int64), torch.tensor([0], dtype=torch.int64), *previous),
                      str(unbound), input_names=names, output_names=output_names, opset_version=15,
                      dynamo=False, export_params=False, do_constant_folding=False,
                      keep_initializers_as_inputs=True, verbose=False)
    graph = onnx.load(str(unbound), load_external_data=False)
    bind_external_weights(graph, wrapper, output.parent, names, output.stem+'.weights.data', minimum_bytes)
    onnx.save_model(graph, str(output))
    onnx.checker.check_model(str(output))
    print('External-weight ONNX checker passed. Independent multi-step numerical validation follows.', flush=True)


def self_test(output):
    import onnxruntime as ort
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig, DynamicCache
    torch.manual_seed(3126)
    config = Gemma3TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                             head_dim=16, sliding_window=4, max_position_embeddings=8,
                             layer_types=['sliding_attention', 'full_attention'],
                             rope_parameters={'sliding_attention': {'rope_type': 'default', 'rope_theta': 10000.0},
                                              'full_attention': {'rope_type': 'default', 'rope_theta': 1000000.0}},
                             attn_logit_softcapping=None, final_logit_softcapping=None,
                             query_pre_attn_scalar=16, use_bidirectional_attention=False,
                             bos_token_id=2, eos_token_id=1, pad_token_id=0)
    config._attn_implementation = 'eager'
    model = Gemma3ForCausalLM(config).float().eval()
    output = Path(output)
    output.parent.mkdir(parents=True)
    export_external(model, 8, output, 1)
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    session = ort.InferenceSession(str(output), sess_options=options, providers=['CPUExecutionProvider'])
    names = [f'past_{kind}_{i}' for i in range(2) for kind in ('k', 'v')]
    out_names = ['logits']+[name.replace('past_', 'present_') for name in names]
    cache = {name: np.zeros((1, 1, 8, 16), dtype=np.float32) for name in names}
    native_cache = DynamicCache(config=config)
    worst = 0.0
    with torch.inference_mode():
        for position, token in enumerate([2, 5, 7, 11, 13, 17, 19, 23]):
            native = model(input_ids=torch.tensor([[token]]), position_ids=torch.tensor([[position]]),
                           past_key_values=native_cache, use_cache=True)
            native_cache = native.past_key_values
            values = session.run(out_names, {'input_ids': np.array([[token]], dtype=np.int64),
                                            'position': np.array([position], dtype=np.int64), **cache})
            delta = float(np.max(np.abs(values[0]-native.logits.numpy())))
            assert delta < 0.0001, 'Tiny Gemma native/ONNX logits mismatch'
            assert int(values[0].argmax()) == int(native.logits.numpy().argmax())
            worst = max(worst, delta)
            for i, name in enumerate(names):
                updated = values[1+i]
                assert np.array_equal(updated[:, :, :position], cache[name][:, :, :position])
                assert np.array_equal(updated[:, :, position+1:], cache[name][:, :, position+1:])
                layer = native_cache.layers[i//2]
                reference = layer.keys if i % 2 == 0 else layer.values
                assert float(np.abs(updated[0, :, position]-reference[0, :, -1].numpy()).max()) < 0.0001
                cache[name] = updated
    (output.parent / 'self-test.json').write_text(json.dumps({'status': 'passed', 'steps': 8,
         'window': 4, 'context': 8, 'max_logit_abs': worst,
         'scope': 'native Gemma hybrid cache versus exported full-cache adapter; external-weight binding, masks, cache slots and final position'}, indent=2)+'\n', encoding='utf-8')
    print('Tiny Gemma external-weight export/native hybrid-cache self-test passed, max abs:', worst, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-len', type=int, default=512)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.self_test:
        self_test(args.output)
        return
    assert args.max_len == 512 and args.model_path
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.float16, device_map='cpu', local_files_only=True,
        trust_remote_code=False, attn_implementation='eager', low_cpu_mem_usage=True).eval()
    assert all(p.dtype == torch.float16 for p in model.parameters())
    assert model.config.num_hidden_layers == 26 and model.config.num_key_value_heads == 1 and model.config.head_dim == 256
    assert model.config.sliding_window == 512
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr(), 'Expected tied embeddings'
    export_external(model, args.max_len, args.output, 1_900_000_000)


if __name__ == '__main__':
    main()
