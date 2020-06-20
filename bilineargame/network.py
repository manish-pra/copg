import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class policy_g(nn.Module):
    def __init__(self):
        super(policy_g, self).__init__()
        # self.actor = nn.Sequential(nn.Linear(state_dim, action_dim),  # 2(self and other's position) * 2(action)
        #                            nn.Softmax(dim =-1))  # 20*2
        self.actor = nn.Parameter(torch.ones(1) * 1)#torch.ones(1,requires_grad=True)

        self.log_std = nn.Parameter(torch.ones(1) * 0.2)

    def forward(self):
        mu = self.actor
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        #dist = Normal(mu,torch.FloatTensor([1]))
        return dist

class policy(nn.Module):
    def __init__(self):
        super(policy, self).__init__()
        # self.actor = nn.Sequential(nn.Linear(state_dim, action_dim),  # 2(self and other's position) * 2(action)
        #                            nn.Softmax(dim =-1))  # 20*2
        self.actor = nn.Parameter(torch.ones(1) * 1)#torch.ones(1,requires_grad=True)

        #self.log_std = nn.Parameter(torch.ones(1) * 1)

    def forward(self):
        mu = self.actor
        #std = torch.FloatTensor(1) #self.log_std.exp().expand_as(mu)
        #dist = Normal(mu,std)
        return mu

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)