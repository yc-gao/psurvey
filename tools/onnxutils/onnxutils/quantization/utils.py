import torch


class LayerObserver(torch.nn.Module):
    def __init__(self, layer, analyser):
        super().__init__()
        self._layer = layer
        self._analyser = analyser

        if hasattr(self._layer, 'onnx_mapping'):
            self.onnx_mapping = self._layer.onnx_mapping

    def target_layer(self):
        return self._layer

    def update(self, vals):
        if not isinstance(vals, (tuple, list)):
            vals = (vals, )
        for name, val in zip(self.onnx_mapping.outputs, vals):
            self._analyser.update({name: val})

    def forward(self, *args):
        ret = self._layer(*args)
        self.update(ret)
        return ret


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
    elif reduction == 'sum':
        return snr.sum()
    elif reduction == 'none':
        return snr
    else:
        raise NotImplementedError


def compute_metric(metric, *args, **kwargs):
    if metric == 'mse':
        val = compute_mse(*args, **kwargs)
    elif metric == 'cosine':
        val = compute_cosine(*args, **kwargs)
    elif metric == 'snr':
        val = compute_snr(*args, **kwargs)
    else:
        raise NotImplementedError
    if val.nelement == 1:
        return val.item()
    return val.detach().cpu().numpy().tolist()
