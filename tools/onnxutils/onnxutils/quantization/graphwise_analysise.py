from tqdm import tqdm

from .utils import LayerObserver, compute_mse, compute_cosine


def graphwise_analyse(model0, model1, dataset, metrics=['mse', 'cosine']):
    metric_to_func = {
        'mse': compute_mse,
        'cosine': compute_cosine,
    }

    model0_recorder = {}
    for name, m in model0.named_children():
        if isinstance(m, LayerObserver):
            continue
        if not hasattr(m, 'onnx_mapping'):
            continue
        setattr(model0, name, LayerObserver(m, model0_recorder))

    model1_recorder = {}
    for name, m in model1.named_children():
        if isinstance(m, LayerObserver):
            continue
        if not hasattr(m, 'onnx_mapping'):
            continue
        setattr(model1, name, LayerObserver(m, model1_recorder))

    results = []
    for idx in tqdm(range(len(dataset))):
        data = dataset[idx]
        model0(*data)
        model1(*data)

        result = {}
        for name in (model0_recorder.keys() | model1_recorder.keys()):
            val0 = model0_recorder.get(name, None)
            val1 = model1_recorder.get(name, None)

            if val0 is None or val1 is None:
                result[name] = None
                continue

            result[name] = [
                metric_to_func[metric](val0, val1)
                for metric in metrics]
        results.append(result)

        model0_recorder.clear()
        model1_recorder.clear()

    return results
