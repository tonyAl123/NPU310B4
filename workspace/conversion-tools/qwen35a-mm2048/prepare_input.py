"""Prepare one image plus text for the fixed 768px / context2048 OM pair; no NPU calls."""
import argparse
import json
from pathlib import Path
import numpy as np
from contract import positions, preprocess_image, prompt_tokens


def vision_input_dtype(interface_path, explicit=None):
    path = Path(interface_path)
    detected = None
    if path.is_file():
        interface = json.loads(path.read_text(encoding='utf-8'))
        assert len(interface['inputs']) == 1
        value = interface['inputs'][0]
        assert value['name'] == 'pixel_values' and value['shape'] == [2304, 1536]
        assert value['dtype'] in ('FLOAT', 'FLOAT16')
        detected = 'float32' if value['dtype'] == 'FLOAT' else 'float16'
    if explicit is not None:
        assert explicit in ('float16', 'float32')
        assert detected is None or explicit == detected, 'Explicit pixel dtype differs from the OM interface'
        return explicit
    assert detected is not None, 'Provide vision-interface.json or explicitly set --vision-input-dtype'
    return detected


def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', default=str(Path(__file__).parent / 'tokenizer'))
    parser.add_argument('--image')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--output', required=True)
    parser.add_argument('--vision-interface', default=str(Path(__file__).with_name('vision-interface.json')))
    parser.add_argument('--vision-input-dtype', choices=['float16', 'float32'])
    args = parser.parse_args()
    tokenizer_path = Path(args.tokenizer)
    config = json.loads((tokenizer_path / 'config.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
    ids = prompt_tokens(tokenizer, args.prompt, bool(args.image))
    assert 0 < args.max_new_tokens and len(ids) + args.max_new_tokens <= 2048, 'Image + prompt + requested answer exceeds context2048'
    pixel_dtype = vision_input_dtype(args.vision_interface, args.vision_input_dtype) if args.image else None
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    np.save(out / 'input_ids.npy', np.array(ids, dtype=np.int64))
    np.save(out / 'rope_positions.npy', positions(ids, config))
    if args.image:
        np.save(out / 'pixel_values.npy', preprocess_image(args.image).astype(pixel_dtype))
    (out / 'request.json').write_text(json.dumps({'input_tokens': len(ids), 'visual_tokens': 576 if args.image else 0,
        'max_new_tokens': args.max_new_tokens, 'context': 2048,
        'image_token_id': config['image_token_id'], 'image_grid_thw': [1, 48, 48] if args.image else None,
        'pixel_values_dtype': pixel_dtype,
        'note': 'Run vision_768.om once, then insert its 576 output vectors at image-token steps of decoder_ctx2048.om; use separate KV slot and MRoPE coordinates.'}, indent=2) + '\n')
    print('Prepared model inputs:', out, '| total input tokens:', len(ids))


if __name__ == '__main__':
    main()
