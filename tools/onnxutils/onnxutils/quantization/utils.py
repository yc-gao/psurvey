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

    @staticmethod
    def observe(model, *args):
        class InnerCls:
            def __init__(self, model, *args):
                self._recorder = {}
                self._model = model
                self._fields = set(args)

            def __enter__(self):
                for name, m in self._model.named_children():
                    if isinstance(m, LayerObserver):
                        continue
                    if not hasattr(m, 'onnx_mapping'):
                        continue
                    if any(x in self._fields for x in m.onnx_mapping.outputs):
                        setattr(self._model, name,
                                LayerObserver(m, self._recorder))
                return self._recorder

            def __exit__(self, exc_type, exc_value, traceback):
                if exc_value is not None:
                    raise exc_value
                LayerObserver.unapply(self._model)
        return InnerCls(model, *args)

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
