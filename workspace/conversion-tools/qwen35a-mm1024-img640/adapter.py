"""Qwen3.5 native blocks with explicit recurrent state, fixed KV and 3-axis MRoPE."""
import argparse
import json
from pathlib import Path
import numpy as np
import onnx
import torch
from torch import nn
from torch.nn import functional as F
from contract import layer_groups, output_names, state_spec
from model_profile import CONTEXT, GRID, PATCH_COUNT, PATCH_WIDTH
from external_weights import bind_external_weights


class FixedAttention:
    def __init__(self, keys, values, position):
        self.keys, self.values = keys, values
        self.mask = (torch.arange(keys.shape[2], device=position.device) == position.reshape(1)).reshape(1, 1, -1, 1)

    def update(self, key, value, layer_idx):
        self.keys = torch.where(self.mask, key, self.keys)
        self.values = torch.where(self.mask, value, self.values)
        return self.keys, self.values


def delta_step(module, hidden, recurrent, conv):
    """One-token recurrence; same math and normalization order as native GatedDeltaNet."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm
    projected = module.in_proj_qkv(hidden).transpose(1, 2)
    next_conv = torch.cat((conv[:, :, 1:], projected), dim=-1)
    mixed = F.silu(F.conv1d(next_conv, module.conv1d.weight, module.conv1d.bias,
                          groups=module.conv_dim)).transpose(1, 2)
    q, k, v = torch.split(mixed, [module.key_dim, module.key_dim, module.value_dim], dim=-1)
    q = l2norm(q.reshape(1, 1, module.num_k_heads, module.head_k_dim))[:, 0].float()
    k = l2norm(k.reshape(1, 1, module.num_k_heads, module.head_k_dim))[:, 0].float()
    v = v.reshape(1, module.num_v_heads, module.head_v_dim).float()
    assert module.num_k_heads == module.num_v_heads
    beta = module.in_proj_b(hidden).sigmoid()[:, 0].float()
    decay = -module.A_log.float().exp() * F.softplus(module.in_proj_a(hidden).float() + module.dt_bias)
    memory = recurrent * decay[:, 0].exp().unsqueeze(-1).unsqueeze(-1)
    correction = (v - (memory * k.unsqueeze(-1)).sum(dim=-2)) * beta.unsqueeze(-1)
    next_recurrent = memory + k.unsqueeze(-1) * correction.unsqueeze(-2)
    value = (next_recurrent * (q / (module.head_k_dim ** .5)).unsqueeze(-1)).sum(dim=-2)
    value = value.to(hidden.dtype).reshape(-1, module.head_v_dim)
    gate = module.in_proj_z(hidden).reshape(-1, module.head_v_dim)
    value = module.norm(value, gate).reshape(1, 1, module.value_dim)
    return module.out_proj(value), next_recurrent, next_conv


class Decoder(nn.Module):
    def __init__(self, model, length):
        super().__init__()
        self.text = model.model.language_model
        self.head = model.lm_head
        self.length = length
        self.cfg = model.config.text_config
        self.linear, self.attention = layer_groups(self.cfg.to_dict())
        assert self.cfg.rope_parameters['rope_type'] == 'default'
        assert self.cfg.linear_num_key_heads == self.cfg.linear_num_value_heads
        self.register_buffer('inverse_frequency', self.text.rotary_emb.inv_freq.detach().float())
        self.sections = self.text.rotary_emb.mrope_section

    def rotary(self, rope_position, dtype):
        frequencies = rope_position.float() * self.inverse_frequency.reshape(1, 1, -1)
        idx = torch.arange(self.inverse_frequency.numel(), device=rope_position.device)
        h = (idx % 3 == 1) & (idx < self.sections[1] * 3)
        w = (idx % 3 == 2) & (idx < self.sections[2] * 3)
        selected = torch.where(h, frequencies[1], torch.where(w, frequencies[2], frequencies[0]))
        embedding = torch.cat((selected, selected), dim=-1).unsqueeze(0)
        scale = self.text.rotary_emb.attention_scaling
        return (embedding.cos() * scale).to(dtype), (embedding.sin() * scale).to(dtype)

    def forward(self, input_ids, position, rope_position, image_embed, is_image, *previous):
        hidden = torch.where(is_image.reshape(1, 1, 1) == 1, image_embed, self.text.embed_tokens(input_ids))
        rotations = self.rotary(rope_position, hidden.dtype)
        slots = torch.arange(self.length, device=position.device)
        mask = torch.where(slots <= position.reshape(1),
                           torch.zeros(self.length, dtype=hidden.dtype, device=hidden.device),
                           torch.full((self.length,), float('-inf'), dtype=hidden.dtype, device=hidden.device))
        mask = mask.reshape(1, 1, 1, self.length)
        n, a = len(self.linear), len(self.attention)
        states, convolutions = list(previous[:n]), list(previous[n:2*n])
        keys, values = list(previous[2*n:2*n+a]), list(previous[2*n+a:])
        di = ai = 0
        for layer in self.text.layers:
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            if layer.layer_type == 'linear_attention':
                mixed, states[di], convolutions[di] = delta_step(layer.linear_attn, normalized, states[di], convolutions[di])
                di += 1
            else:
                cache = FixedAttention(keys[ai], values[ai], position)
                mixed, _ = layer.self_attn(normalized, position_embeddings=rotations,
                                           attention_mask=mask, past_key_values=cache)
                keys[ai], values[ai] = cache.keys, cache.values
                ai += 1
            hidden = residual + mixed
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        logits = self.head(self.text.norm(hidden))
        return (logits, *states, *convolutions, *keys, *values)


class Vision(nn.Module):
    def __init__(self, visual, grid, output_dtype=None):
        super().__init__()
        self.patch = visual.patch_embed.proj
        self.blocks, self.merger = visual.blocks, visual.merger
        self.output_dtype = output_dtype
        grid_tensor = torch.tensor([grid], dtype=torch.int64)
        with torch.no_grad():
            absolute = visual.fast_pos_embed_interpolate(grid_tensor)
            rotary = visual.rot_pos_emb(grid_tensor).reshape(int(np.prod(grid)), -1)
            angles = torch.cat((rotary, rotary), dim=-1)
        self.register_buffer('absolute', absolute.detach().to(self.patch.weight.dtype))
        self.register_buffer('cosine', angles.cos().detach())
        self.register_buffer('sine', angles.sin().detach())
        self.register_buffer('cu', torch.tensor([0, int(np.prod(grid))], dtype=torch.int32))

    def forward(self, pixel_values):
        # A non-overlapping patch-sized Conv3D equals a linear projection of each flattened patch.
        hidden = F.linear(pixel_values.to(self.patch.weight.dtype),
                          self.patch.weight.reshape(self.patch.out_channels, -1), self.patch.bias)
        hidden = hidden + self.absolute
        for block in self.blocks:
            hidden = block(hidden, cu_seqlens=self.cu, position_embeddings=(self.cosine, self.sine))
        # Preserve the public FP16 bridge even when the vision internals use FP32.
        return self.merger(hidden).to(self.output_dtype or pixel_values.dtype)


def export_graph(wrapper, examples, names, outputs, directory):
    directory = Path(directory)
    directory.mkdir(parents=True)
    graph_path = directory / 'model.onnx'
    unbound = directory / 'unbound.onnx'
    wrapper.eval()
    with torch.no_grad():
        torch.onnx.export(wrapper, examples, str(unbound), input_names=names, output_names=outputs,
                          opset_version=15, dynamo=False, export_params=False,
                          keep_initializers_as_inputs=True, do_constant_folding=False)
    graph = onnx.load(str(unbound), load_external_data=False)
    bind_external_weights(graph, wrapper, directory, names, 'model.weights.data', 1)
    onnx.save(graph, str(graph_path))
    onnx.checker.check_model(str(graph_path))
    return graph_path


def decoder_examples(model, length):
    cfg = model.config.text_config
    dtype = next(model.parameters()).dtype
    floating = 'float16' if dtype == torch.float16 else 'float32'
    states = [torch.zeros(shape, dtype=getattr(torch, kind))
              for _, shape, kind in state_spec(cfg.to_dict(), length, floating)]
    names = ['input_ids', 'position', 'rope_position', 'image_embed', 'is_image']
    names += [name for name, _, _ in state_spec(cfg.to_dict(), length)]
    inputs = (torch.ones((1, 1), dtype=torch.int64), torch.zeros(1, dtype=torch.int64),
              torch.zeros((3, 1, 1), dtype=torch.int64), torch.zeros((1, 1, cfg.hidden_size), dtype=dtype),
              torch.zeros(1, dtype=torch.int64), *states)
    return inputs, names


def load_model(path, dtype):
    from transformers import Qwen3_5ForConditionalGeneration
    model, info = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(path), dtype=dtype, device_map='cpu', trust_remote_code=False,
        local_files_only=True, attn_implementation='eager', low_cpu_mem_usage=True, output_loading_info=True)
    assert not any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')), info
    model.eval()
    assert model.config.text_config._attn_implementation == 'eager'
    assert model.config.vision_config._attn_implementation == 'eager'
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['vision', 'decoder', 'selftest'])
    parser.add_argument('--model')
    parser.add_argument('--output', required=True)
    parser.add_argument('--vision-fp32', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.mode == 'selftest':
        from selftest import run
        run(Path(args.output))
        return
    model = load_model(args.model, torch.float16)
    cfg = model.config
    assert (cfg.text_config.hidden_size, cfg.text_config.num_hidden_layers, cfg.text_config.vocab_size) == (2048, 24, 248320)
    assert (cfg.vision_config.patch_size, cfg.vision_config.spatial_merge_size, cfg.vision_config.out_hidden_size) == (16, 2, 2048)
    if args.mode == 'vision':
        if args.vision_fp32:
            model.model.visual.float()
        input_dtype = torch.float32 if args.vision_fp32 else torch.float16
        export_graph(Vision(model.model.visual, GRID, output_dtype=torch.float16),
                     (torch.zeros((PATCH_COUNT, PATCH_WIDTH), dtype=input_dtype),), ['pixel_values'], ['image_features'], args.output)
    else:
        examples, names = decoder_examples(model, CONTEXT)
        export_graph(Decoder(model, CONTEXT), examples, names, output_names(cfg.to_dict()), args.output)


if __name__ == '__main__':
    main()
