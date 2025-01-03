import torch


class LayerObserver(torch.nn.Module):
    @staticmethod
    def apply(model, recorder):
        for name, m in model.named_children():
            if isinstance(m, LayerObserver):
                continue
            if not hasattr(m, 'onnx_mapping'):
                continue
            setattr(model, name, LayerObserver(m, recorder))
        return LayerObserver(model, recorder)

    @staticmethod
    def unapply(model):
        if isinstance(model, LayerObserver):
            model = model.target_layer()
        for name, m in model.named_children():
            if isinstance(m, LayerObserver):
                setattr(model, name, m.target_layer())
        return model

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
