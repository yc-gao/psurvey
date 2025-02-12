#!/usr/bin/env python3
import argparse

from torchvision import models
from torch.utils.data import DataLoader

from imagenet_pipeline import ImageNetPipeline


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path')
    return parser.parse_args()


def main():
    options = parse_options()

    torch_model = models.resnet18(models.ResNet18_Weights.DEFAULT)
    dataloader = DataLoader(ImageNetPipeline.get_dataset(
        options.dataset_path), batch_size=16)

    ImageNetPipeline.train(torch_model, dataloader)
    ImageNetPipeline.eval(torch_model, dataloader)


if __name__ == "__main__":
    main()
