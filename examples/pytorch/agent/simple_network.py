import torch

class SimpleNet(torch.nn.Module):

    def __init__(self, insize, outsize, activation=lambda x: x):
        super().__init__()
        self.insize = insize
        self.outsize = outsize
        self.activation = activation

        self.fc1 = torch.nn.Linear(self.insize, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.head = torch.nn.Linear(64, self.outsize)

    def forward(self, input):
        x = torch.nn.functional.relu(self.fc1(input))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.activation(self.head(x))

class DoubleInputNet(torch.nn.Module):

    def __init__(self, firstinsize, secondinsize, outsize, activation=lambda x: x):
        super().__init__()
        self.firstinsize = firstinsize
        self.secondinsize = secondinsize
        self.outsize = outsize
        self.activation = activation

        self.fc1_1 = torch.nn.Linear(firstinsize, 64)
        self.fc1_2 = torch.nn.Linear(secondinsize, 64)
        self.fc2 = torch.nn.Linear(128, 64)
        self.head = torch.nn.Linear(64, self.outsize)

    def forward(self, firstin, secondin):
        x1 = torch.nn.functional.relu(self.fc1_1(firstin))
        x2 = torch.nn.functional.relu(self.fc1_2(secondin))
        x = torch.cat([x1, x2], dim=1)
        x = torch.nn.functional.relu(self.fc2(x))
        return self.activation(self.head(x))