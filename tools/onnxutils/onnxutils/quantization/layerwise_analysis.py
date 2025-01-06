from tqdm import tqdm

from .layer_observer import LayerObserver, compute_metrics


# def layerwise_analyse(model, quantized_model, dataloader, metrics=['snr', 'mse', 'cosine'], **kwargs):
#     model_recorder = {}
#     model = LayerObserver.apply(model, model_recorder)
#
#     stats = []
#     for data in tqdm(dataloader):
#         for k, val in zip(model.onnx_mapping.inputs, data):
#             model_recorder[k] = val
#         model(*data)
#
#         batch_stat = []
#         for layer in quantized_model.children():
#             if not hasattr(layer, 'onnx_mapping'):
#                 continue
#             args = tuple(model_recorder.get(x, None)
#                          for x in layer.onnx_mapping.inputs)
#             if None in args:
#                 continue
#             vals = layer(*args)
#             if not isinstance(vals, (tuple, list)):
#                 vals = (vals,)
#             for name, val in zip(layer.onnx_mapping.outputs, vals):
#                 real = model_recorder.get(name, None)
#                 pred = val
#                 if pred is None or real is None:
#                     continue
#
#                 tensor_stat = compute_metrics(metrics, real, pred, **kwargs)
#                 tensor_stat['name'] = name
#                 batch_stat.append(tensor_stat)
#
#         stats.append(batch_stat)
#         model_recorder.clear()
#
#     model = LayerObserver.unapply(model)
#
#     return stats
