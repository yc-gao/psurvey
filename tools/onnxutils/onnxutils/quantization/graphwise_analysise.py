from tqdm import tqdm

from .utils import LayerObserver, compute_metrics


def graphwise_analyse(model0, model1, dataloader, metrics=['snr', 'mse', 'cosine'], **kwargs):
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

        result = []
        for name in (model0_recorder.keys() | model1_recorder.keys()):
            val0 = model0_recorder.get(name, None)
            val1 = model1_recorder.get(name, None)

            if val0 is None or val1 is None:
                continue

            stat = compute_metrics(metrics, val0, val1, **kwargs)
            stat['name'] = name
            result.append(stat)

        results.append(result)
        model0_recorder.clear()
        model1_recorder.clear()

    for name, m in model0.named_children():
        if isinstance(m, LayerObserver):
            setattr(model0, name, m.target_layer())
    for name, m in model1.named_children():
        if isinstance(m, LayerObserver):
            setattr(model1, name, m.target_layer())

    return results
