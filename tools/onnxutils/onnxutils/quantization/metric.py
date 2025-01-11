import copy

import torch


def do_compute(real, pred, fn):
    assert type(real) is type(pred)

    if isinstance(real, torch.Tensor):
        return fn(real, pred)

    if isinstance(real, (list, tuple)):
        return [__compute(a, b, fn) for a, b in zip(real, pred)]

    if isinstance(real, dict):
        return {
            __compute(real.get(key, None), pred.get(key, None), fn)
            for key in real.keys() | pred.keys()
        }

    raise NotImplementedError


def compute_mse(real, pred, reduction='none'):
    def inner_kernel(real, pred):
        assert real.shape == pred.shape
        if real.dim() < 2:
            real = real.reshape(-1, 1)
            pred = pred.reshape(-1, 1)

        real = real.flatten(start_dim=1).float()
        pred = pred.flatten(start_dim=1).float()

        metrics = torch.pow(real - pred, 2).mean(dim=-1)

        if reduction == 'none':
            return metrics
        if reduction == 'max':
            return metrics.max()
        if reduction == 'min':
            return metrics.min()
        if reduction == 'mean':
            return metrics.mean()
        if reduction == 'sum':
            return metrics.sum()

        raise NotImplementedError
    return do_compute(real, pred, inner_kernel)


def compute_cosine(real, pred, reduction='none'):
    def inner_kernel(real, pred):
        assert real.shape == pred.shape
        if real.dim() < 2:
            real = real.reshape(-1, 1)
            pred = pred.reshape(-1, 1)

        real = real.flatten(start_dim=1).float()
        pred = pred.flatten(start_dim=1).float()

        metrics = torch.cosine_similarity(real, pred, dim=-1)

        if reduction == 'none':
            return metrics
        if reduction == 'max':
            return metrics.max()
        if reduction == 'min':
            return metrics.min()
        if reduction == 'mean':
            return metrics.mean()
        if reduction == 'sum':
            return metrics.sum()

        raise NotImplementedError
    return do_compute(real, pred, inner_kernel)


def compute_snr(real, pred, eps=1e-7, reduction='none'):
    def inner_kernel(real, pred):
        assert real.shape == pred.shape
        if real.dim() < 2:
            real = real.reshape(-1, 1)
            pred = pred.reshape(-1, 1)

        real = real.flatten(start_dim=1).float()
        pred = pred.flatten(start_dim=1).float()

        metrics = (
            torch.pow(real - pred, 2) / (torch.pow(real, 2) + eps)
        ).sum(dim=-1)

        if reduction == 'none':
            return metrics
        if reduction == 'max':
            return metrics.max()
        if reduction == 'min':
            return metrics.min()
        if reduction == 'mean':
            return metrics.mean()
        if reduction == 'sum':
            return metrics.sum()

        raise NotImplementedError
    return do_compute(real, pred, inner_kernel)


def compute_metrics(real, pred, metrics=['cosine', 'snr', 'mse'], *args, **kwargs):
    def inner_kernel(real, pred):
        metric2func = {
            'cosine': compute_cosine,
            'snr': compute_snr,
            'mse': compute_mse
        }
        return {
            m: metric2func[m](real, pred, *args, **kwargs)
            for m in metrics
        }
    return do_compute(real, pred, inner_kernel)
