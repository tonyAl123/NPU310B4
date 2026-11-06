"""Fixed image profile and explicit hybrid-state interface; no model patches."""
import numpy as np


def layer_groups(config):
    cfg = config.get('text_config', config)
    return ([i for i, kind in enumerate(cfg['layer_types']) if kind == 'linear_attention'],
            [i for i, kind in enumerate(cfg['layer_types']) if kind == 'full_attention'])


def state_spec(config, length, floating='float16'):
    cfg = config.get('text_config', config)
    linear, attention = layer_groups(config)
    kh, vh = cfg['linear_num_key_heads'], cfg['linear_num_value_heads']
    kd, vd = cfg['linear_key_head_dim'], cfg['linear_value_head_dim']
    assert kh == vh
    conv = 2 * kh * kd + vh * vd
    result = []
    for i in range(len(linear)):
        result.append((f's_past_{i}', [1, vh, kd, vd], 'float32'))
    for i in range(len(linear)):
        result.append((f'c_past_{i}', [1, conv, cfg['linear_conv_kernel_dim']], floating))
    for kind in ('k', 'v'):
        for i in range(len(attention)):
            result.append((f'{kind}_past_{i}', [1, cfg['num_key_value_heads'], length, cfg['head_dim']], floating))
    return result


def zero_state(config, length, floating='float16'):
    return {name: np.zeros(shape, dtype=dtype) for name, shape, dtype in state_spec(config, length, floating)}


def output_names(config):
    return ['logits'] + [name.replace('_past_', '_pres_') for name, _, _ in state_spec(config, 1)]


def letterbox(image, size=768):
    from PIL import Image, ImageOps
    image = ImageOps.exif_transpose(image).convert('RGB')
    scale = min(size / image.width, size / image.height)
    dimensions = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = image.resize(dimensions, Image.Resampling.BICUBIC)
    canvas = Image.new('RGB', (size, size), (128, 128, 128))
    canvas.paste(resized, ((size - dimensions[0]) // 2, (size - dimensions[1]) // 2))
    return canvas


def image_patches(rgb):
    """Single still image, 16px patches, 2x2 spatial merge, duplicated temporal frame."""
    rgb = np.asarray(rgb)
    assert rgb.shape == (768, 768, 3) and rgb.dtype == np.uint8
    normalized = (rgb.astype(np.float32) / np.float32(255) - np.float32(.5)) / np.float32(.5)
    cells = normalized.transpose(2, 0, 1).reshape(3, 24, 2, 16, 24, 2, 16)
    cells = cells.transpose(1, 4, 2, 5, 0, 3, 6)
    patches = np.repeat(cells[..., None, :, :], 2, axis=-3).reshape(2304, 1536)
    return np.ascontiguousarray(patches)


def preprocess_image(path):
    from PIL import Image
    with Image.open(path) as image:
        rgb = np.asarray(letterbox(image), dtype=np.uint8)
    return image_patches(rgb)


def positions(ids, config, grid=(1, 48, 48)):
    """MRoPE coordinates are distinct from the monotonically increasing KV slot."""
    ids = np.asarray(ids, dtype=np.int64)
    indices = np.flatnonzero(ids == config['image_token_id'])
    if not len(indices):
        return np.broadcast_to(np.arange(len(ids))[None, None, :], (3, 1, len(ids))).copy()
    gh, gw = grid[1] // 2, grid[2] // 2
    assert grid[0] == 1 and len(indices) == gh * gw
    start, end = int(indices[0]), int(indices[-1]) + 1
    assert np.array_equal(indices, np.arange(start, end)), 'Only one contiguous image is supported'
    result = np.empty((3, 1, len(ids)), dtype=np.int64)
    result[:, 0, :start] = np.arange(start)
    result[0, 0, start:end] = start
    result[1, 0, start:end] = start + np.repeat(np.arange(gh), gw)
    result[2, 0, start:end] = start + np.tile(np.arange(gw), gh)
    result[:, 0, end:] = start + max(gh, gw) + np.arange(len(ids) - end)
    return result


def prompt_tokens(tokenizer, prompt, image=False):
    content = [{'type': 'image'}, {'type': 'text', 'text': prompt}] if image else prompt
    text = tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
                                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if image:
        assert text.count('<|image_pad|>') == 1
        text = text.replace('<|image_pad|>', '<|image_pad|>' * 576)
    return tokenizer.encode(text, add_special_tokens=False)
