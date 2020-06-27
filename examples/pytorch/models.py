import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
from torch.autograd import Variable

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device="cpu"):
        super(Critic, self).__init__()
        self.device = device
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1).to(self.device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 3e-4, device="cpu"):
        super(Actor, self).__init__()
        self.device = device
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.head = nn.Linear(hidden_size, output_size)

        
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        bsize = state.shape[0]
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.head(x)

        steer = x[:,0].reshape(-1,1)
        accel = x[:,1].reshape(-1,1)
        brake = x[:,2].reshape(-1,1)

        steer = torch.tanh(steer)
        accel = torch.sigmoid(accel).reshape(-1,1)
        brake = torch.sigmoid(brake).reshape(-1,1)

        return torch.cat((steer,accel,brake), 1)

