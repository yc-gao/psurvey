#!/usr/bin/env python3
import argparse
import functools
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)


from torchvision import models, datasets, transforms


class transform_iterator:
    def __init__(self, iter, *transforms):
        self.iter = iter
        self.transforms = transforms

    def __len__(self):
        return len(self.iter)

    def __iter__(self):
        class inner:
            def __init__(self, iter, transforms):
                self.iter = iter
                self.transforms = transforms

            def __next__(self):
                return functools.reduce(lambda item, f: f(item), (next(self.iter), ) + self.transforms)
        return inner(iter(self.iter), self.transforms)


class ImageNetPipeline:
    @staticmethod
    def get_dataloader(root_dir, batch_size=64, image_size=224):
        dataset = datasets.ImageFolder(root_dir, transforms.Compose([
            transforms.Resize(image_size + 24),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ]))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)

    @staticmethod
    def train(model, dataloader, optimizer, loss_fn):
        size = len(dataloader)
        # model.train()
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)

            optimizer.zero_grad()
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            if batch % 10 == 0:
                print(f"loss: {loss.item():>7f}  [{batch:>5d}/{size:>5d}]")

    @staticmethod
    def eval(model, dataloader):
        total_count = 0
        total_correct = 0
        # model.eval()
        with torch.no_grad():
            for data, label in tqdm(dataloader):
                pred = torch.argmax(model(data), dim=1)
                correct = (label == pred).sum()

                total_count = total_count + data.shape[0]
                total_correct = total_correct + correct
        return total_correct * 1.0 / total_count, total_count, total_correct

    @staticmethod
    def calibrate(model, dataloader):
        # model.eval()
        with torch.no_grad():
            for data, _ in tqdm(dataloader):
                model(data)


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight')
    parser.add_argument(
        '--dataset-path', help="Imagenet eval image", type=str, required=True)
    parser.add_argument('-o', '--output')
    return parser.parse_args()


def main():
    options = parse_options()

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    dataloader = transform_iterator(
        ImageNetPipeline.get_dataloader(options.dataset_path), lambda items: [x.to(device) for x in items])
    example_inputs = (next(iter(dataloader))[0],)

    float_model = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V2)
    if options.weight:
        float_model.load_state_dict(torch.load(options.weight))
    float_model = float_model.eval().to(device)

    exported_model = torch.export.export_for_training(
        float_model, example_inputs).module()

    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config())
    prepared_model = prepare_pt2e(exported_model, quantizer)

    ImageNetPipeline.calibrate(prepared_model, dataloader)

    quantized_model = convert_pt2e(prepared_model)

    acc, _, _ = ImageNetPipeline.eval(float_model, dataloader)
    print(f"origin model, acc: {acc * 100:.4f}%")
    acc, _, _ = ImageNetPipeline.eval(quantized_model, dataloader)
    print(f"quantized model, acc: {acc * 100:.4f}%")

    if options.output:
        # WARNING: can't export onnx format
        torch.onnx.export(
            quantized_model, example_inputs, options.output)


if __name__ == "__main__":
    main()
