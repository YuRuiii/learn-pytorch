import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn import datasets as skdatasets
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import numpy as np

class cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d1 = nn.Conv2d(1, 16, (5,5))
        self.conv2d2 = nn.Conv2d(16, 16, (5,5))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(6400, 6400)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(6400, 1000)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(1000, 10)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        # print("forward", x)
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x

def get_data():
    # image_trans = ToTensor()
    transforms_MNIST = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.1307,), std = (0.3081,))
        ]
    )
    target_trans = lambda x: nn.functional.one_hot(torch.arange(10))[x]
    trainset = datasets.MNIST('mnist_train', train=True, download=True,  transform=transforms_MNIST, target_transform=target_trans)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = datasets.MNIST('mnist_test', train=False, download=True, transform=transforms_MNIST, target_transform=target_trans)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    return trainloader, testloader


def train(train_dataloader, test_dataloader, model, loss_fn, optimizer):
    # model.train()
    writer = SummaryWriter("./log")
    total_step = 0
    for epoch in tqdm(range(100)):
        model.train()
        for _, (X, y) in tqdm(enumerate(train_dataloader)):
            total_step += 1
            pred = model(X)
            # print(pred, y.float())
            loss = loss_fn(pred, y.float())
            writer.add_scalar('Loss/train', loss, total_step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        cr = test(test_dataloader, model)
        writer.add_scalar('Correct_rate', cr, total_step)

def test(test_dataloader, model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad(): # means no need to calculate & save grad in the whole graph
        for X, y in test_dataloader:
            pred = model(X)
            correct += 1 if torch.argmax(pred) == y else 0
            total += 1
    return correct/total
    
if __name__ == "__main__":
    train_dataloader, test_dataloader = get_data()
    model = cnn()
    loss_fn = lambda pred, y: nn.functional.cross_entropy(pred, y, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(train_dataloader, test_dataloader, model, loss_fn, optimizer)
    # test(test_dataloader, model)