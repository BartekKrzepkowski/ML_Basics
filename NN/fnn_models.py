import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, dims):
        super(Net, self).__init__()
        self.fc = torch.nn.ModuleList([nn.Linear(dim_in, dim_out)\
                                       for dim_in, dim_out in zip(dims[:-1], dims[1:])])

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        for layer in self.fc[:-1]:
            x = torch.relu(layer(x))
        return self.fc[-1](x)


class LinearRegression(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(LinearRegression, self).__init__()
        self.i2o = nn.Linear(x_dim, y_dim, bias=True)
        self.loss = nn.MSELoss()

    def forward(self, x):
        return self.i2o(x)


class MnistFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1024, 2500), nn.BatchNorm1d(2500), nn.ReLU(),
                                 nn.Linear(2500, 2000), nn.BatchNorm1d(2000), nn.ReLU(),
                                 nn.Linear(2000, 1500), nn.BatchNorm1d(1500), nn.ReLU(),
                                 nn.Linear(1500, 1000), nn.BatchNorm1d(1000), nn.ReLU(),
                                 nn.Linear(1000, 500), nn.BatchNorm1d(500), nn.ReLU(),
                                 nn.Linear(500, 10))

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.net(x)
        return x