import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn import datasets as skdatasets
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import numpy as np



class TypoNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 10)
        self.linear2 = nn.Linear(10, 10)
        self.linear3 = nn.Linear(10, 3)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        # print("forward", x)
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        x = self.tanh(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x

def get_data():
    iris = skdatasets.load_iris()
    X_train, X_test, y_train_raw, y_test = train_test_split(iris.data, iris.target, random_state=104, test_size=0.25, shuffle=True)
    y_train = []
    for _, y in enumerate(y_train_raw):
        y_train.append({0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]}[y]) # encode it to be one-hot

    tensor_x = torch.Tensor(X_train) # create dataset in numpy: 1. transform it to tensor type; 2. use TensorDataset to transform it to dataset type
    tensor_y = torch.Tensor(y_train)
    train_dataset = TensorDataset(tensor_x,tensor_y)
    train_dataloader = DataLoader(train_dataset,batch_size=16, shuffle=True)
    tensor_x = torch.Tensor(X_test)
    tensor_y = torch.Tensor(y_test)
    test_dataset = TensorDataset(tensor_x,tensor_y)
    test_dataloader = DataLoader(test_dataset)
    return train_dataset, test_dataset, train_dataloader, test_dataloader

def train(train_dataloader, test_dataloader, model, loss_fn, optimizer):
    # model.train()
    writer = SummaryWriter("./log")
    total_step = 0
    for epoch in tqdm(range(100)):
        model.train()
        for _, (X, y) in enumerate(train_dataloader):
            total_step += 1
            pred = model(X)
            loss = loss_fn(pred, y)
            writer.add_scalar('Loss/train', loss, total_step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        cr = test(test_dataloader, model)
        writer.add_scalar('Correct_Rate', cr, total_step)

def test(test_dataloader, model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad(): # means no need to calculate & save grad in the whole graph
        for X, y in test_dataloader:
            pred = model(X)
            # print(list(X), list(pred), int(y), int(torch.argmax(pred)))
            correct += 1 if torch.argmax(pred) == y else 0
            total += 1
    return correct/total
    # return '%.3f'%(correct/total)
    # print("prediction correct rate:", '%.3f'%(correct/total))
    
if __name__ == "__main__":
    train_dataset, test_dataset, train_dataloader, test_dataloader = get_data()
    model = TypoNetwork()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(train_dataloader, test_dataloader, model, loss_fn, optimizer)
    # test(test_dataloader, model)