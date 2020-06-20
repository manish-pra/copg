import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class policy_temp(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(policy_temp, self).__init__()
        self.actor = nn.Parameter(torch.ones(1) * 2)

    def forward(self, state):
        mu = self.actor#(state)
        std = torch.FloatTensor([1])
        dist = Normal(mu,std)
        return dist

class policy1(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(policy1, self).__init__()
        self.actor = nn.Sequential(nn.Linear(state_dim, 1, bias=False))  # 20*2

        #self.actor = nn.Parameter(torch.ones(1) * 5)
        self.log_std = nn.Parameter(torch.ones(1) * 0.1)

        self.apply(init_weights)

    def forward(self, state):
        mu = self.actor(state)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu,std)
        return dist

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.1, std=0.0001)
        #nn.init.constant_(m.bias, 0.1)

class policy2(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(policy2, self).__init__()
        self.actor = nn.Sequential(nn.Linear(state_dim, 1, bias=False))  # 20*2

        #self.actor = nn.Parameter(torch.ones(1) * 5)
        self.log_std = nn.Parameter(torch.ones(1) * 0.1)

        self.apply(init_weights2)

    def forward(self, state):
        mu = self.actor(state)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu,std)
        return dist

def init_weights2(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=-0.1, std=0.0001)#-0.6

class critic(nn.Module):
    def __init__(self, state_dim):
        super(critic, self).__init__()

        self.critic = nn.Sequential(nn.Linear(state_dim, 24),  # 84*50
                                    nn.Tanh(),
                                    nn.Linear(24, 24),  # 50*20
                                    nn.Tanh(),
                                    nn.Linear(24, 1))  # 20*2

        #self.apply(init_weights)

    def forward(self, state):
        value = self.critic(state)
        return value

# class critic(nn.Module):
#     def __init__(self, state_dim):
#         super(critic, self).__init__()
#         self.critic = nn.Sequential(nn.Linear(state_dim, 1),  # 2(self and other's position) * 2(action)
#                                    nn.Tanh(),
#                                     )  # 20*2
#     def forward(self, state):
#         val = self.critic(state)
#         return val
