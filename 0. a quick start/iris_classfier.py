from enum import EnumMeta
import enum
from pickletools import optimize
from turtle import forward
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn import datasets as skdatasets
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm



class TypoNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 10)
        self.linear2 = nn.Linear(10, 3)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        # print("forward", x)
        x1 = self.linear1(x)
        x2 = self.linear2(x1)
        x3 = self.softmax(x2)
        return x3
    
if __name__ == "__main__":
    # get data
    iris = skdatasets.load_iris()
    X_train, X_test, y_train_raw, y_test = train_test_split(iris.data, iris.target, random_state=104, test_size=0.25, shuffle=True)
    y_train = []
    for _, y in enumerate(y_train_raw):
        y_train.append({0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]}[y]) # encode it to be one-hot

    tensor_x = torch.Tensor(X_train) # create dataset in numpy: 1. transform it to tensor type; 2. use TensorDataset to transform it to dataset type
    tensor_y = torch.Tensor(y_train)
    train_dataset = TensorDataset(tensor_x,tensor_y)
    train_dataloader = DataLoader(train_dataset)
    tensor_x = torch.Tensor(X_test)
    tensor_y = torch.Tensor(y_test)
    test_dataset = TensorDataset(tensor_x,tensor_y)
    test_dataloader = DataLoader(test_dataset)

    model = TypoNetwork()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    # train

    for _, (X, y) in tqdm(enumerate(train_dataloader)):
        # print("here", X, y)
        pred = model(X)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # test
    correct, total = 0, 0
    with torch.no_grad(): # means no need to calculate & save grad in the whole graph
        for X, y in tqdm(test_dataloader):
            pred = model(X)
            print(X, y, pred, torch.argmax(pred))
            correct += 1 if torch.argmax(pred) == y else 0
            total += 1
    print("prediction correct rate:", '%.3f'%(correct/total))