import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from models import *
from utils import *

class DDPGagent(nn.Module):
    def __init__(self, env,params,insize=23, device="cpu"):
        super().__init__()
        # Params
        self.num_states = insize
        self.num_actions = env.action_space.shape[0]
        self.gamma = params.gamma
        self.tau = params.tau
        self.device = device

        self.hidden_size = 256
        # Networks
        self.actor = Actor(self.num_states, self.hidden_size, self.num_actions, device=self.device).to(self.device)
        self.actor_target = Actor(self.num_states, self.hidden_size, self.num_actions, device=self.device).to(self.device)
        self.critic = Critic(self.num_states + self.num_actions, self.hidden_size, self.num_actions, device=self.device).to(self.device)
        self.critic_target = Critic(self.num_states + self.num_actions, self.hidden_size, self.num_actions, device=self.device).to(self.device)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Training
        self.memory = Memory(params.buffersize)        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=params.lrvalue)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=params.lrpolicy)
    
    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0)).to(self.device)
        action = self.actor.forward(state)
        action = action.detach().cpu().numpy()[0]

        return action
    
    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
    
        # Critic loss        
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        
        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def _totorch(self, container, dtype):
        if isinstance(container[0], torch.Tensor):
            tensor = torch.stack(container)
        else:
            tensor = torch.tensor(container, dtype=dtype)
        return tensor#.to(self.device)

    def to(self, device):
        self.device = device
        super().to(device)