from tqdm import tqdm

from .utils import LayerObserver, compute_metric


def graphwise_analyse(model0, model1, dataloader, metrics=['mse', 'cosine']):
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
    for data in tqdm(dataloader):
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
                compute_metric(metric, val0, val1)
                for metric in metrics]
        results.append(result)

        model0_recorder.clear()
        model1_recorder.clear()

    return results
