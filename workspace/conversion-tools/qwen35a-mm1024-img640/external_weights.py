import json
from pathlib import Path
import onnx
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


