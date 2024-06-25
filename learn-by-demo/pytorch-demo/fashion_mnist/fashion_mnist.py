#!/usr/bin/env python3

import torch
from torch import nn
from torch.utils.data import DataLoader

import models
import utils


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

batch_size = 64


def main():
    train_dataloader, test_dataloader = [DataLoader(
        x, batch_size=batch_size) for x in utils.make_dataset()]

    model = models.NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        utils.train(model, loss_fn, optimizer, train_dataloader, device)
        utils.test(model, loss_fn, test_dataloader, device)
    print("Done!")

    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")


if __name__ == "__main__":
    main()
