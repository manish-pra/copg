import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, '..')
import numpy as np
from bilineargame.bilinear_game import bilinear
from torch.utils.tensorboard import SummaryWriter

from bilineargame.network import policy_g as policy

folder_location = 'tensorboard/bilinear/'
experiment_name = 'gda/'
directory = '../' + folder_location + experiment_name + 'model'

if not os.path.exists(directory):
    os.makedirs(directory)
writer = SummaryWriter('../' + folder_location + experiment_name + 'data')

p1 = policy()
p2 = policy()

optim_p1 = torch.optim.SGD(p1.parameters(), lr=0.1)
optim_p2 = torch.optim.SGD(p2.parameters(), lr=0.1)

batch_size = 2000
num_episode = 400

env = bilinear()

for t_eps in range(num_episode):
    mat_action = []

    mat_state1 = []
    mat_reward1 = []
    mat_done = []

    mat_state2 = []
    mat_reward2 = []
    state, _, _, _, _ = env.reset()
    #data_collection
    for i in range(batch_size):
        dist1 = p1()
        action1 = dist1.sample() # it will learn probablity for actions and output 0,1,2,3 as possibility of actions

        dist2 = p2()
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
    writer.add_scalar('Action/Agent1', torch.mean(action_both[:,0]), t_eps)
    writer.add_scalar('Action/agent2', torch.mean(action_both[:,1]), t_eps)

    # writer.add_scalar('Mean/agent1', dist1.mean, t_eps)
    # writer.add_scalar('Mean/agent2', dist2.mean, t_eps)
    # writer.add_scalar('Variance/agent1', dist1.variance, t_eps)
    # writer.add_scalar('Variance/agent2', dist2.variance, t_eps)

    val1_p = torch.stack(mat_reward1).transpose(0,1)

    pi_a1_s = p1()
    log_probs1 = pi_a1_s.log_prob(action_both[:,0])
    objective = -log_probs1 * (val1_p)
    ob = objective.mean()
    optim_p1.zero_grad()
    ob.backward()
    optim_p1.step()

    val2_p = -val1_p
    pi_a2_s = p2()
    log_probs2 = pi_a2_s.log_prob(action_both[:, 1])

    objective2 = -log_probs2 * (val2_p)
    optim_p2.zero_grad()
    ob2 = objective2.mean()
    ob2.backward()
    optim_p2.step()

    # writer.add_scalar('Objective/Agent1', ob, t_eps)
    # writer.add_scalar('Objective/agent2', ob2, t_eps)


    if t_eps%100==0:
        print(t_eps)
        torch.save(p1.state_dict(),
                   '../' + folder_location + experiment_name + 'model/agent1_' + str(
                       t_eps) + ".pth")
        torch.save(p2.state_dict(),
                   '../' + folder_location + experiment_name + 'model/agent2_' + str(
                       t_eps) + ".pth")