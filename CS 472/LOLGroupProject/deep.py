import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super().__init__()  # must call this at the start every time
        self.fc1 = nn.Linear(15, 64)  # fully connected linear layer, takes in the size of the image (28*28)
        self.fc2 = nn.Linear(64, 64)  # takes in 64 nodes (second argument from first layer)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 2)  # finally, 10 output classes (digits 0-9)

    def forward(self, x):  # defines a forward step through the model, needed for output = net(input)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    df1 = pd.read_csv("data/dataset.csv")
    df2 = pd.read_csv("data/christopher.csv")
    df3 = pd.read_csv("data/mason.csv")
    df = df1.append(df2).append(df3)
    cols = list(df.columns.values)
    cols.pop(cols.index("won?"))
    df = df[cols + ["won?"]]

    init_data = df.to_numpy()

    scaler = MinMaxScaler()
    scaler.fit(init_data)
    norm_data = scaler.transform(init_data)

    data = init_data[:, 0:-1]
    labels = init_data[:, -1]  # .reshape(-1, 1)
    # labels = labels.astype(int)

    data = data.tolist()
    labels = labels.tolist()
    new_labels = []
    for label in labels:
        if label:
            new_labels.append([0])  # [1, 0])
        else:
            new_labels.append([1])  # [0, 1])

    for i in range(len(data)):
        data[i] = torch.Tensor(data[i])
    for i in range(len(new_labels)):
        new_labels[i] = torch.Tensor(new_labels[i])

    # data = torch.Tensor(data)
    # labels = torch.Tensor(labels)

    # data_loader = (data, labels)

    # data_set = TensorDataset(data, labels)
    # data_loader = DataLoader(data_set, batch_size=1)

    net = Net()

    # measures our loss -- how far off the target we are
    loss_function = nn.MSELoss()

    # lr is the learning rate -- how quickly the model should make adjustments
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Train
    net.train()  # we are training the model
    for epoch in range(30):  # 8 full passes over the data
        correct = 0
        for i in range(len(data)):  # `data` is a batch of data
            X = data[i].view(-1, 15)
            y = new_labels[i]  # .view(-1, 2)
            # X, y = data  # X is the batch of features, y is the batch of targets
            net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
            output = net(X)  # pass in the reshaped batch (recall they are 28x28 atm)
            if torch.argmax(output) == y.float():
                correct += 1
            loss = loss_function(output, y)  # calc and grab the loss value
            # also could use loss_function(output, y)  which we defined earlier
            loss.backward()  # apply this loss backwards through the network's parameters
            optimizer.step()  # attempt to optimize weights to account for loss/gradients
        print("Loss:", loss.item(), " |  Acc:", correct / len(data))  # print loss
