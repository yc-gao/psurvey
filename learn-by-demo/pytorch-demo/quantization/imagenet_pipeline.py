#!/usr/bin/env python3
import functools
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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
    def get_example_inputs():
        return (torch.randn(1, 3, 224, 224),)

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

        model.train()
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

        model.eval()
        with torch.no_grad():
            for data, label in tqdm(dataloader):
                pred = torch.argmax(model(data), dim=1)
                correct = (label == pred).sum()

                total_count = total_count + data.shape[0]
                total_correct = total_correct + correct
        return total_correct * 1.0 / total_count

    @staticmethod
    def calibrate(model, dataloader):
        model.eval()
        with torch.no_grad():
            for data, _ in tqdm(dataloader):
                model(data)
