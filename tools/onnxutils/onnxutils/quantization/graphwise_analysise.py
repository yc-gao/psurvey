from tqdm import tqdm

from .layer_observer import LayerObserver
from .metric import compute_metrics


def graphwise_analyse(model0, model1, dataloader, metrics=['snr', 'mse', 'cosine'], **kwargs):
    model0_recorder = {}
    model0 = LayerObserver.apply(model0, model0_recorder)

    model1_recorder = {}
    model1 = LayerObserver.apply(model1, model1_recorder)

    stats = []
    for data in tqdm(dataloader):
        model0(*data)
        model1(*data)

        stat = compute_metrics(
            model0_recorder,
            model1_recorder,
            metrics,
            **kwargs)

        stats.append(stat)
        model0_recorder.clear()
        model1_recorder.clear()

    model0 = LayerObserver.unapply(model0)
    model1 = LayerObserver.unapply(model1)

    return stats
