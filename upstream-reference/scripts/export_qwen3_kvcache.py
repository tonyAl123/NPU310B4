#!/usr/bin/env python3
"""
Export Qwen3-0.6B with KV Cache (fixed-size buffers) for Atlas 200I DK A2.

KV buffer per layer: (1, num_kv_heads=8, max_len, head_dim=128)
Inputs:  58 = input_ids + position + 28×(K+V)
Outputs: 57 = logits + 28×(present_K+present_V)
"""

import argparse, os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm, Qwen3Attention, Qwen3DecoderLayer,
    apply_rotary_pos_emb, eager_attention_forward,
)


# ── KV Cache insert (ONNX-friendly: where + arange) ────────────────
def insert_to_cache(cache: torch.Tensor, new_kv: torch.Tensor,
                    position: torch.Tensor) -> torch.Tensor:
    """
    cache:  (1, H, max_len, D)   pre-allocated
    new_kv: (1, H, 1, D)         single-token K or V
    position: scalar tensor like tensor([3])
    Returns: (1, H, max_len, D) with new_kv written at position
    """
    L = cache.shape[2]
    idx = torch.arange(L, dtype=torch.int64, device=cache.device)
    mask = idx.unsqueeze(0).unsqueeze(0).unsqueeze(-1) == position.view(1, 1, 1, 1)
    return torch.where(mask, new_kv, cache)


# ── Causal mask (all-tensor, no .item()) ───────────────────────────
def make_attn_mask(max_len: int, position: torch.Tensor,
                   dtype=torch.float16) -> torch.Tensor:
    """
    position: scalar tensor like tensor([3])
    Returns: (1, 1, 1, max_len), 0=attend, -inf=mask
    Masks key positions > position (causal decode).
    """
    L = max_len
    idx = torch.arange(L, dtype=torch.int64)
    # True where key position > current query position
    mask = idx.unsqueeze(0).unsqueeze(0) > position  # (1, 1, L)
    bias = torch.full((1, 1, 1, L), float("-inf"), dtype=dtype)
    bias = bias.masked_fill(~mask.unsqueeze(2), 0.0)  # (1, 1, 1, L)
    return bias


# ── Monkey-patch: Qwen3Attention ───────────────────────────────────
_original_attention_forward = Qwen3Attention.forward

def _patched_attention_forward(self, hidden_states, position_embeddings,
                                attention_mask, past_k=None, past_v=None,
                                position=None, **kwargs):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Insert into fixed cache (no GQA repeat — eager_attention_forward does it)
    new_past_k = past_k
    new_past_v = past_v
    if past_k is not None and position is not None:
        key_states = insert_to_cache(past_k, key_states, position)
        value_states = insert_to_cache(past_v, value_states, position)
        new_past_k = key_states    # updated full K
        new_past_v = value_states  # updated full V

    attn_output, attn_weights = eager_attention_forward(
        self, query_states, key_states, value_states,
        attention_mask, dropout=0.0, scaling=self.scaling,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, new_past_k, new_past_v


# ── Monkey-patch: Qwen3DecoderLayer ────────────────────────────────
_original_decoder_forward = Qwen3DecoderLayer.forward

def _patched_decoder_forward(self, hidden_states, attention_mask,
                              position_embeddings, position_ids,
                              past_k=None, past_v=None, position=None, **kwargs):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    hidden_states, _, new_past_k, new_past_v = self.self_attn(
        hidden_states=hidden_states, attention_mask=attention_mask,
        position_embeddings=position_embeddings, position_ids=position_ids,
        past_k=past_k, past_v=past_v, position=position,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states, new_past_k, new_past_v


# ── Wrapper ────────────────────────────────────────────────────────
class KVCacheWrapper(nn.Module):
    def __init__(self, model, max_len: int):
        super().__init__()
        self.model = model
        self.max_len = max_len
        self.num_layers = model.config.num_hidden_layers
        self.num_kv_heads = model.config.num_key_value_heads
        self.head_dim = getattr(model.config, "head_dim",
                                model.config.hidden_size // model.config.num_attention_heads)

    def forward(self, input_ids, position, *kv_past):
        L = self.max_len
        attn_mask = make_attn_mask(L, position)

        inputs_embeds = self.model.model.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        pos_ids = position.unsqueeze(0)
        position_embeddings = self.model.model.rotary_emb(hidden_states, pos_ids)

        kv_present = []
        layers = self.model.model.layers[:self.model.config.num_hidden_layers]

        for i, layer in enumerate(layers):
            hidden_states, pk, pv = layer(
                hidden_states,
                attention_mask=attn_mask,
                position_embeddings=position_embeddings,
                position_ids=pos_ids,
                past_k=kv_past[2 * i],
                past_v=kv_past[2 * i + 1],
                position=position,
            )
            kv_present.append(pk)
            kv_present.append(pv)

        hidden_states = self.model.model.norm(hidden_states)
        logits = self.model.lm_head(hidden_states)

        return (logits, *kv_present)


# ── Main ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="model/Qwen3-0.6B")
    parser.add_argument("--output", default="qwen3_kvcache_max256.onnx")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--validate", action="store_true", default=True)
    args = parser.parse_args()

    N = args.max_len
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16,
        device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True,
    ).eval()

    cfg = model.config
    NL = cfg.num_hidden_layers
    NKV = cfg.num_key_value_heads
    HD = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    print(f"  layers={NL} kv_heads={NKV} head_dim={HD} max_len={N}")

    # Apply monkey-patches
    Qwen3Attention.forward = _patched_attention_forward
    Qwen3DecoderLayer.forward = _patched_decoder_forward

    wrapper = KVCacheWrapper(model, N)

    input_names = ["input_ids", "position"]
    for i in range(NL):
        input_names.append(f"past_k_{i}")
        input_names.append(f"past_v_{i}")

    output_names = ["logits"]
    for i in range(NL):
        output_names.append(f"present_k_{i}")
        output_names.append(f"present_v_{i}")

    print(f"  Inputs: {len(input_names)}, Outputs: {len(output_names)}")

    dummy_in = torch.ones((1, 1), dtype=torch.int64)
    dummy_pos = torch.tensor([0], dtype=torch.int64)
    kv_shape = (1, NKV, N, HD)
    dummy_kv = [torch.zeros(kv_shape, dtype=torch.float16) for _ in range(NL * 2)]

    print(f"Exporting to {args.output}...")
    torch.onnx.export(
        wrapper, (dummy_in, dummy_pos, *dummy_kv),
        args.output,
        input_names=input_names, output_names=output_names,
        opset_version=15, do_constant_folding=True,
        dynamo=False, verbose=False,
    )
    print(f"  Size: {os.path.getsize(args.output) / 1024 / 1024:.1f} MB")

    if args.validate:
        print("Validating with ONNX Runtime...")
        import onnx
        import onnxruntime as ort

        onnx.checker.check_model(args.output)
        print("  ONNX checker: PASS")

        sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
        feed = {
            "input_ids": np.ones((1, 1), dtype=np.int64),
            "position": np.array([0], dtype=np.int64),
        }
        for i in range(NL):
            feed[f"past_k_{i}"] = np.zeros(kv_shape, dtype=np.float16)
            feed[f"past_v_{i}"] = np.zeros(kv_shape, dtype=np.float16)

        outputs = sess.run(None, feed)
        logits = outputs[0]
        print(f"  logits: shape={logits.shape}, range=[{logits.min():.4f},{logits.max():.4f}]")
        assert logits.shape == (1, 1, cfg.vocab_size)
        assert np.isfinite(logits).all()

        for i in range(NL):
            assert outputs[1 + 2*i].shape == kv_shape, f"k{i} bad shape"
            assert outputs[2 + 2*i].shape == kv_shape, f"v{i} bad shape"
        print(f"  All {NL} K/V tensors: PASS")

    print("Done.")


if __name__ == "__main__":
    main()
