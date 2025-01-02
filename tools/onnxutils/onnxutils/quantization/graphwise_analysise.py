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
    model0 = LayerObserver(model0, model0_recorder)

    model1_recorder = {}
    for name, m in model1.named_children():
        if isinstance(m, LayerObserver):
            continue
        if not hasattr(m, 'onnx_mapping'):
            continue
        setattr(model1, name, LayerObserver(m, model1_recorder))
    model1 = LayerObserver(model1, model1_recorder)

    stats = []
    for data in tqdm(dataloader):
        model0(*data)
        model1(*data)

        batch_stat = []
        for name in (model0_recorder.keys() | model1_recorder.keys()):
            val0 = model0_recorder.get(name, None)
            val1 = model1_recorder.get(name, None)

            if val0 is None or val1 is None:
                continue

            tensor_stat = compute_metrics(metrics, val0, val1, **kwargs)
            tensor_stat['name'] = name
            batch_stat.append(tensor_stat)

        stats.append(batch_stat)
        model0_recorder.clear()
        model1_recorder.clear()

    model0 = model0.target_layer()
    for name, m in model0.named_children():
        if isinstance(m, LayerObserver):
            setattr(model0, name, m.target_layer())

    model1 = model1.target_layer()
    for name, m in model1.named_children():
        if isinstance(m, LayerObserver):
            setattr(model1, name, m.target_layer())

    return stats
