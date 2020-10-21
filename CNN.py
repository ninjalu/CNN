
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# SET UP TRAINING VISUALISATION
from torch.utils.tensorboard import SummaryWriter

# SET UP TRAINING VISUALISATION
writer = SummaryWriter(log_dir='../runs')


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=10,
                            kernel_size=3),   # output 10*26*26
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=10,
                            out_channels=4,
                            kernel_size=3),  # output is 4*24*24
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=3),  # output is 16*8*8=1024
            torch.nn.Flatten(),
            torch.nn.Linear(2304, 10),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


train_dataset = datasets.MNIST(root='MNIST-data',
                               transform=transforms.ToTensor(),
                               train=True,
                               download=True)

train_data, val_data = torch.utils.data.random_split(
    train_dataset, [int(0.7*len(train_dataset)), int(0.3*len(train_dataset))])

train_loader = torch.utils.data.dataloader.DataLoader(
    train_data, batch_size=16, shuffle=True)

val_loader = torch.utils.data.dataloader.DataLoader(
    val_data, batch_size=16, shuffle=True)

cnn_minst = CNN()


def train(model, epoch, train_loader):
    losses = []
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    loss_ce = F.cross_entropy
    acc_train = []
    acc_val = []

    for epoch in range(epoch):
        for idx, batch in enumerate(train_loader):
            X, y = batch
            y_pred = model(X)
            loss = loss_ce(y_pred, y)
            # print(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(idx, loss.item())
            a_train = np.mean(np.array(torch.argmax(y_pred) == y))

        # COMPUTE LOSS ON VALIDATION SET

        for val_batch in val_loader:
            x, y = val_batch  # (x, y)
            # makew a prediction
            y_pred = model(x)
            a_val = np.mean(np.array(torch.argmax(y_pred) == y))

        acc_train.append(a_train)
        acc_val.append(a_val)

        losses.append(loss)

    return losses, acc_train, acc_val


losses, acc_train, acc_val = train(cnn_minst, 5, train_loader)
# X = []
# for i in range(len(y)):
#     X.append(i)
plt.plot(losses)
plt.plot(acc_train, c='r', label='Training')
plt.plot(acc_val, c='b', label='Validation')
plt.show()
# plt.figure(figsize=(20, 15))plt.plot(X, Y, c='r', label='Label')
# plt.scatter(X, y_hat, c='b', label='Estimation', marker='x')
