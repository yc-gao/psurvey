#!/usr/bin/env python3

import matplotlib.pyplot as plt

from tinygrad import Tensor, nn
from tinygrad.nn.datasets import mnist

Tensor.manual_seed(42)


class Model:
    def __init__(self):
        self.l1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.l2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.l3 = nn.Linear(1600, 10)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.l1(x).relu().max_pool2d((2, 2))
        x = self.l2(x).relu().max_pool2d((2, 2))
        return self.l3(x.flatten(1).dropout(0.5))


def do_train(model, X_train, Y_train):
    optim = nn.optim.Adam(nn.state.get_parameters(model))
    with Tensor.train():
        for i in range(1000):
            samples = Tensor.randint(64, high=X_train.shape[0])
            X, Y = X_train[samples], Y_train[samples]
            optim.zero_grad()
            loss = model(X).sparse_categorical_crossentropy(Y).backward()
            optim.step()
            if i % 100 == 0:
                print(f"step {i:4d}, loss {loss.item():.2f}")


def do_eval(model, X_test, Y_test):
    with Tensor.train(False):
        acc = (model(X_test).argmax(axis=1) == Y_test).sum() / X_test.shape[0]
        print(f"eval, acc {acc.item()*100.:.2f}%")
        samples = Tensor.randint(8, high=X_test.shape[0])
        X, Y = X_test[samples], Y_test[samples]
        pred = model(X).argmax(axis=1)
        # X: (8, 1, 28, 28)
        # Y: (8,)
        # pred: (8,)
        _, axs = plt.subplots(4, 2)
        for i in range(8):
            axs[i % 4, i // 4].set_axis_off()
            axs[i % 4, i // 4].imshow(X[i].squeeze().numpy(), cmap='gray')
            axs[i % 4, i //
                4].set_title(f"pred {pred[i].item()} real {Y[i].item()}")
        plt.show()


def main():
    X_train, Y_train, X_test, Y_test = mnist()
    model = Model()
    do_train(model, X_train, Y_train)
    do_eval(model, X_test, Y_test)


if __name__ == "__main__":
    main()
