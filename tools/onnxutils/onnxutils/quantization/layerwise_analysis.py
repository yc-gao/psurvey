from tqdm import tqdm

from .utils import LayerObserver, compute_mse, compute_cosine


def layerwise_analyse(model, quantized_model, dataset, metrics=['mse', 'cosine']):
    metric_to_func = {
        'mse': compute_mse,
        'cosine': compute_cosine,
    }

    model_recorder = {}
    for name, m in model.named_children():
        if isinstance(m, LayerObserver):
            continue
        if not hasattr(m, 'onnx_mapping'):
            continue
        setattr(model, name, LayerObserver(m, model_recorder))

    results = []
    for idx in tqdm(range(len(dataset))):
        data = dataset[idx]
        for k, val in zip(model.onnx_mapping.inputs, data):
            model_recorder[k] = val
        model(*data)

        result = {}
        for layer in quantized_model.children():
            if not hasattr(layer, 'onnx_mapping'):
                continue
            args = tuple(model_recorder.get(x, None)
                         for x in layer.onnx_mapping.inputs)
            if None in args:
                continue
            vals = layer(*args)
            if not isinstance(vals, (tuple, list)):
                vals = (vals,)
            for name, val in zip(layer.onnx_mapping.outputs, vals):
                real = model_recorder.get(name, None)
                pred = val
                if pred is None or real is None:
                    result[name] = None
                    continue
                result[name] = [
                    metric_to_func[metric](real, pred)
                    for metric in metrics]
        results.append(result)
    return results
