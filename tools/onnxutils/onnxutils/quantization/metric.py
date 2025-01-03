import copy

import torch


def compute_mse(real, pred, reduction='none'):
    assert real.shape == pred.shape
    if real.ndim == 1:
        real = real.unsqueeze(0)
        pred = pred.unsqueeze(0)

    real = real.flatten(start_dim=1).float()
    pred = pred.flatten(start_dim=1).float()

    mse = torch.pow(real - pred, 2).mean(dim=-1)
    if reduction == 'mean':
        return mse.mean()
    if reduction == 'max':
        return mse.max()
    if reduction == 'min':
        return mse.min()
    elif reduction == 'sum':
        return mse.sum()
    elif reduction == 'none':
        return mse
    else:
        raise NotImplementedError


def compute_cosine(real, pred, reduction='none'):
    assert real.shape == pred.shape
    if real.ndim == 1:
        real = real.unsqueeze(0)
        pred = pred.unsqueeze(0)

    real = real.flatten(start_dim=1).float()
    pred = pred.flatten(start_dim=1).float()

    cosine_sim = torch.cosine_similarity(real, pred, dim=-1)
    if reduction == 'mean':
        return cosine_sim.mean()
    if reduction == 'max':
        return cosine_sim.max()
    if reduction == 'min':
        return cosine_sim.min()
    elif reduction == 'sum':
        return cosine_sim.sum()
    elif reduction == 'none':
        return cosine_sim
    else:
        raise NotImplementedError


def compute_snr(real, pred, reduction='none'):
    assert real.shape == pred.shape
    if real.ndim == 1:
        real = real.unsqueeze(0)
        pred = pred.unsqueeze(0)

    real = real.flatten(start_dim=1).float()
    pred = pred.flatten(start_dim=1).float()

    signal_power = torch.pow(real, 2).sum(dim=-1)
    noise_power = torch.pow(real - pred, 2).sum(dim=-1)

    snr = (noise_power) / (signal_power + 1e-7)
    if reduction == 'mean':
        return snr.mean()
    if reduction == 'max':
        return snr.max()
    if reduction == 'min':
        return snr.min()
    elif reduction == 'sum':
        return snr.sum()
    elif reduction == 'none':
        return snr
    else:
        raise NotImplementedError


def compute_metrics(recorder0, recorder1, metrics, *args, **kwargs):
    def compute(val0, val1, metric, *args, **kwargs):
        if metric == 'mse':
            val = compute_mse(val0, val1, *args, **kwargs)
        elif metric == 'cosine':
            val = compute_cosine(val0, val1, *args, **kwargs)
        elif metric == 'snr':
            val = compute_snr(val0, val1, *args, **kwargs)
        else:
            raise NotImplementedError
        if val.nelement == 1:
            return val.item()
        return val.detach().cpu().numpy().tolist()

    stat = []
    for name in (recorder0.keys() & recorder1.keys()):
        val0 = recorder0.get(name, None)
        val1 = recorder1.get(name, None)
        if val0 is None or val1 is None:
            continue

        tmp = {metric: compute(val0, val1, metric, *args, **kwargs)
               for metric in metrics}
        tmp['field'] = name
        stat.append(tmp)

    return stat


def print_stats(stats, sorted_metric=None, reversed_order=None):
    stats = copy.deepcopy(stats)
    for stat in stats:
        if sorted_metric is not None and reversed_order is not None:
            stat = sorted(
                stat, key=lambda x: x[sorted_metric], reverse=reversed_order)
        for tensor_stat in stat:
            name = tensor_stat.pop('field')
            print(name, tensor_stat)
