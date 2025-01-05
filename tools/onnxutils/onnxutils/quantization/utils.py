import torch


class LayerObserver(torch.nn.Module):
    @staticmethod
    def apply(model, recorder):
        for name, m in model.named_children():
            if isinstance(m, LayerObserver):
                continue
            m = LayerObserver.apply(m, recorder)
            setattr(model, name, m)

        if hasattr(model, 'onnx_mapping'):
            return LayerObserver(model, recorder)
        return model

    @staticmethod
    def unapply(model):
        if isinstance(model, LayerObserver):
            model = model.target_layer()

        for name, m in model.named_children():
            m = LayerObserver.unapply(m)
            setattr(model, name, m)

        return model

    @staticmethod
    def observe(model, *args):
        class InnerCls:
            def __init__(self, model, *args):
                self._recorder = {}
                self._model = model
                self._fields = set(args)

            def update(self, record):
                self._recorder.update({
                    k: v for k, v in record.items() if k in self._fields
                })

            def model(self):
                return self._model

            def value(self, name):
                return self._recorder.get(name, None)

            def __enter__(self):
                self._model = LayerObserver.apply(self._model, self)
                return self

            def __exit__(self, *args):
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
