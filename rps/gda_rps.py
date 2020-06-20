import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, '..')
from torch.distributions import Categorical
import numpy as np
from rps.rps_game import rps_game
from torch.utils.tensorboard import SummaryWriter

from rps.network import policy1, policy2

import time


folder_location = 'tensorboard/rps/'
experiment_name = 'gda/'
directory = '../' + folder_location + experiment_name + 'model'

if not os.path.exists(directory):
    os.makedirs(directory)
writer = SummaryWriter('../' + folder_location + experiment_name + 'data')

p1 = policy1()
p2 = policy2()

optim_p1 = torch.optim.SGD(p1.parameters(), lr=0.5)
optim_p2 = torch.optim.SGD(p2.parameters(), lr=0.5)

batch_size = 1000
num_episode = 1000

env = rps_game()

for t_eps in range(num_episode):
    mat_action = []

    mat_state1 = []
    mat_reward1 = []
    mat_done = []

    mat_state2 = []
    mat_reward2 = []
    state, _, _, _, _ = env.reset()
    #data_collection
    for i in range(1000):
        pi1 = p1()
        dist1 = Categorical(pi1)
        action1 = dist1.sample()

        pi2 = p2()
        dist2 = Categorical(pi2)
        action2 = dist2.sample()
        action = np.array([action1, action2])

        state = np.array([0,0])
        mat_state1.append(torch.FloatTensor(state))
        mat_state2.append(torch.FloatTensor(state))
        mat_action.append(torch.FloatTensor(action))

        state, reward1, reward2, done, _ = env.step(action)
        mat_reward1.append(torch.FloatTensor([reward1]))
        mat_reward2.append(torch.FloatTensor([reward2]))
        mat_done.append(torch.FloatTensor([1 - done]))


    action_both = torch.stack(mat_action)
    writer.add_scalar('Entropy/Agent1', dist1.entropy().data, t_eps)
    writer.add_scalar('Entropy/agent2', dist2.entropy().data, t_eps)

    writer.add_scalar('Action/Agent1', torch.mean(action_both[:,0]), t_eps)
    writer.add_scalar('Action/agent2', torch.mean(action_both[:,1]), t_eps)

    val1_p = torch.stack(mat_reward1).transpose(0,1)

    if val1_p.size(0)!=1:
        raise 'error'

    pi_a1_s = p1()
    dist_pi1 = Categorical(pi_a1_s)
    action_both = torch.stack(mat_action)
    log_probs1 = dist_pi1.log_prob(action_both[:,0])

    objective = -log_probs1 * (val1_p)
    ob = objective.mean()
    optim_p1.zero_grad()
    ob.backward()
    optim_p1.step()

    val2_p = -val1_p
    pi_a2_s = p2()
    dist_pi2 = Categorical(pi_a2_s)
    log_probs2 = dist_pi2.log_prob(action_both[:,1])

    objective2 = -log_probs2 * (val2_p)
    optim_p2.zero_grad()
    ob2 = objective2.mean()
    ob2.backward()
    optim_p2.step()

    # if t_eps%100 ==0:
    #     for p in p1.parameters():
    #         print('p1', p)
    #     for p in p2.parameters():
    #         print('p2', p)

    writer.add_scalar('Agent1/sm1', pi1.data[0], t_eps)
    writer.add_scalar('Agent1/sm2', pi1.data[1], t_eps)
    writer.add_scalar('Agent1/sm3', pi1.data[2], t_eps)
    writer.add_scalar('Agent2/sm1', pi2.data[0], t_eps)
    writer.add_scalar('Agent2/sm2', pi2.data[1], t_eps)
    writer.add_scalar('Agent2/sm3', pi2.data[2], t_eps)

    ed_time = time.time()

    if t_eps%100==0:
        print(t_eps)
        torch.save(p1.state_dict(),
                   '../' + folder_location + experiment_name + 'model/agent1_' + str(
                       t_eps) + ".pth")
        torch.save(p2.state_dict(),
                   '../' + folder_location + experiment_name + 'model/agent2_' + str(
                       t_eps) + ".pth")