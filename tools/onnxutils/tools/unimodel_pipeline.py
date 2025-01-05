from tqdm import tqdm

from torchvision import datasets, transforms

import torch
from torch import nn
from torch.utils.data import DataLoader


class ImageNetPipeline:
    @staticmethod
    def get_dataloader(root_dir):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        train_dataset = datasets.ImageFolder(
            root=root_dir, transform=transform)
        dataloader = DataLoader(train_dataset, batch_size=64)
        return dataloader

    @staticmethod
    def eval(model, dataloader, device):
        total = 0
        correct = 0
        model.to(device)
        with torch.no_grad():
            for images, labels in tqdm(dataloader):
                images, labels = images.to(device), labels.to(device)
                pred = torch.argmax(model(images), dim=1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        print(f'Accuracy: {100 * correct / total}%')

    @staticmethod
    def train(model, dataloader, device):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        model.to(device)
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            pred = model(images)

            optimizer.zero_grad()
            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()
