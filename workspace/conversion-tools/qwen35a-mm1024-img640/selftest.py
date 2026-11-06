"""Small native-vs-ONNX proof of both subgraphs, MRoPE and hybrid state transitions."""
import json
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from contract import output_names, positions, zero_state
from native import arrays


def run(root):
    from transformers import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig, Qwen3_5ForConditionalGeneration
    from adapter import Decoder, Vision, decoder_examples, export_graph
    torch.manual_seed(3502048)
    text = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                           num_hidden_layers=2, layer_types=['linear_attention', 'full_attention'],
                           num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                           linear_num_key_heads=2, linear_num_value_heads=2,
                           linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
                           max_position_embeddings=16, full_attention_interval=2,
                           rope_parameters={'rope_type': 'default', 'rope_theta': 10000.0,
                                            'partial_rotary_factor': .5, 'mrope_interleaved': True,
                                            'mrope_section': [2, 1, 1]}, tie_word_embeddings=True)
    vision = Qwen3_5VisionConfig(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
                                patch_size=2, temporal_patch_size=2, spatial_merge_size=2,
                                num_position_embeddings=16, out_hidden_size=32)
    cfg = Qwen3_5Config(text_config=text.to_dict(), vision_config=vision.to_dict(),
                       image_token_id=60, vision_start_token_id=61, vision_end_token_id=62,
                       video_token_id=63, tie_word_embeddings=True)
    cfg._attn_implementation = 'eager'
    cfg.text_config._attn_implementation = 'eager'
    cfg.vision_config._attn_implementation = 'eager'
    model = Qwen3_5ForConditionalGeneration(cfg).float().eval()
    pixel = torch.randn((16, 24))
    export_graph(Vision(model.model.visual, (1, 4, 4)), (pixel,), ['pixel_values'], ['image_features'], root / 'vision')
    examples, names = decoder_examples(model, 16)
    export_graph(Decoder(model, 16), examples, names, output_names(cfg.to_dict()), root / 'decoder')
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    # Exercise the repair profile: FP16 public image interface, FP32 vision internals.
    half_pixel = pixel.half()
    export_graph(Vision(model.model.visual, (1, 4, 4)), (half_pixel,),
                 ['pixel_values'], ['image_features'], root / 'vision-fp32')
    repaired_encoder = ort.InferenceSession(str(root / 'vision-fp32/model.onnx'),
        sess_options=options, providers=['CPUExecutionProvider'])
    with torch.inference_mode():
        reference_half = model.model.visual(half_pixel.float(), grid_thw=torch.tensor([[1, 4, 4]])).pooler_output.half().numpy()
    repaired_features = repaired_encoder.run(None, {'pixel_values': half_pixel.numpy()})[0]
    mixed_vision_error = float(np.abs(repaired_features.astype(np.float32) - reference_half.astype(np.float32)).max())
    assert repaired_features.dtype == np.float16 and mixed_vision_error < .0002, mixed_vision_error
    del repaired_encoder
    export_graph(Vision(model.model.visual, (1, 4, 4), output_dtype=torch.float16), (pixel,),
                 ['pixel_values'], ['image_features'], root / 'vision-fp32-input')
    fp32_input_encoder = ort.InferenceSession(str(root / 'vision-fp32-input/model.onnx'),
        sess_options=options, providers=['CPUExecutionProvider'])
    with torch.inference_mode():
        fp32_input_reference = model.model.visual(pixel, grid_thw=torch.tensor([[1, 4, 4]])).pooler_output.half().numpy()
    fp32_input_features = fp32_input_encoder.run(None, {'pixel_values': pixel.numpy()})[0]
    fp32_input_error = float(np.abs(fp32_input_features.astype(np.float32) - fp32_input_reference.astype(np.float32)).max())
    assert fp32_input_features.dtype == np.float16 and fp32_input_error < .0002, fp32_input_error
    del fp32_input_encoder
    encoder = ort.InferenceSession(str(root / 'vision/model.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
    decoder = ort.InferenceSession(str(root / 'decoder/model.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
    ids = np.array([1, 61, 60, 60, 60, 60, 62, 5, 7, 11, 13, 17, 19, 23, 29, 31])
    rope = positions(ids, cfg.to_dict(), (1, 4, 4))
    actual_rope, _ = model.model.get_rope_index(input_ids=torch.from_numpy(ids[None]),
                                               image_grid_thw=torch.tensor([[1, 4, 4]]),
                                               mm_token_type_ids=torch.from_numpy((ids[None] == 60).astype(np.int64)))
    assert np.array_equal(rope, actual_rope.numpy())
    state, native_cache, worst = zero_state(cfg.to_dict(), 16, 'float32'), None, 0.
    with torch.inference_mode():
        expected_features = model.model.visual(pixel, grid_thw=torch.tensor([[1, 4, 4]])).pooler_output
        actual_features = encoder.run(None, {'pixel_values': pixel.numpy()})[0]
        vision_error = float(np.abs(actual_features - expected_features.numpy()).max())
        assert vision_error < .0002, vision_error
        image_index = 0
        for pos, token in enumerate(ids):
            image = token == 60
            value = actual_features[image_index:image_index+1].reshape(1, 1, 32) if image else np.zeros((1, 1, 32), np.float32)
            expected_embed = expected_features[image_index].reshape(1, 1, 32) if image else model.model.language_model.embed_tokens(torch.tensor([[token]]))
            if image:
                image_index += 1
            native = model.model.language_model(inputs_embeds=expected_embed,
                position_ids=torch.cat((torch.tensor([[[pos]]]), torch.from_numpy(rope[:, :, pos:pos+1])), dim=0),
                past_key_values=native_cache, use_cache=True)
            native_cache = native.past_key_values
            expected_logits = model.lm_head(native.last_hidden_state).numpy()
            values = decoder.run(output_names(cfg.to_dict()), {
                'input_ids': np.array([[token]], np.int64), 'position': np.array([pos], np.int64),
                'rope_position': rope[:, :, pos:pos+1], 'image_embed': value,
                'is_image': np.array([int(image)], np.int64), **state})
            error = float(np.abs(values[0] - expected_logits).max())
            worst = max(worst, error)
            assert error < .0002 and values[0].argmax() == expected_logits.argmax(), error
            expected = arrays(native_cache, cfg.to_dict(), 16, floating='float32')
            for (name, previous), updated in zip(state.items(), values[1:]):
                assert np.isfinite(updated).all()
                assert np.max(np.abs(updated - expected[name])) < .0002, name
                if name.startswith(('k_', 'v_')):
                    assert np.array_equal(previous[:, :, :pos], updated[:, :, :pos])
                    assert np.array_equal(previous[:, :, pos+1:], updated[:, :, pos+1:])
                state[name] = updated
    (root / 'self-test.json').write_text(json.dumps({'status': 'passed', 'steps': 16, 'context': 16,
        'image_tokens': 4, 'vision_max_abs': vision_error, 'logits_max_abs': worst,
        'fp32_vision_fp16_interface_max_abs': mixed_vision_error,
        'fp32_vision_input_fp16_output_max_abs': fp32_input_error,
        'scope': 'Native multimodal MRoPE, linearized patch projection, image/word selection, GDN S/conv, KV and final slot'}, indent=2) + '\n')
