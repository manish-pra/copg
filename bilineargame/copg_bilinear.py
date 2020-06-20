import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, '..')
from copg_optim import CoPG
import numpy as np
from bilineargame.bilinear_game import bilinear
from torch.utils.tensorboard import SummaryWriter
from bilineargame.network import policy_g as policy
import time

folder_location = 'tensorboard/bilinear/'
experiment_name = 'copg/'
directory = '../' + folder_location + experiment_name + 'model'

if not os.path.exists(directory):
    os.makedirs(directory)
writer = SummaryWriter('../' + folder_location + experiment_name + 'data')

p1 = policy()
p2 = policy()

# for p in p1.parameters():
#     print(p)

optim = CoPG(p1.parameters(),p2.parameters(), lr = 0.1)

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
        action1 = dist1.sample()

        dist2 = p2()
        action2 = dist2.sample()
        action = np.array([action1, action2])

        state = np.array([0, 0])
        mat_state1.append(torch.FloatTensor(state))
        mat_state2.append(torch.FloatTensor(state))
        mat_action.append(torch.FloatTensor(action))

        state, reward1, reward2, done, _ = env.step(action)

        mat_reward1.append(torch.FloatTensor([reward1]))
        mat_reward2.append(torch.FloatTensor([reward2]))

        mat_done.append(torch.FloatTensor([1 - done]))

    action_both = torch.stack(mat_action)

    val1_p = torch.stack(mat_reward1).transpose(0, 1)

    if val1_p.size(0) != 1:
        raise 'error'

    pi_a1_s = p1()
    action_both = torch.stack(mat_action)
    writer.add_scalar('Action/Agent1', torch.mean(action_both[:,0]), t_eps)
    writer.add_scalar('Action/agent2', torch.mean(action_both[:,1]), t_eps)

    log_probs1 = pi_a1_s.log_prob(action_both[:, 0])

    pi_a2_s = p2()
    log_probs2 = pi_a2_s.log_prob(action_both[:, 1])

    objective = log_probs1 * log_probs2 * (val1_p)
    if objective.size(0) != 1:
        raise 'error'

    ob = objective.mean()

    s_log_probs1 = log_probs1.clone()
    s_log_probs2 = log_probs2.clone()

    for i in range(1, log_probs1.size(0)):
        s_log_probs1[i] = torch.add(s_log_probs1[i - 1], log_probs1[i])
        s_log_probs2[i] = torch.add(s_log_probs2[i - 1], log_probs2[i])

    objective2 = s_log_probs1 * log_probs2 * (val1_p)
    ob2 = objective2.mean()

    objective3 = log_probs1 * s_log_probs2 * (val1_p)
    ob3 = objective3.mean()

    lp1 = log_probs1 * val1_p
    lp1 = lp1.mean()
    lp2 = log_probs2 * val1_p
    lp2 = lp2.mean()
    optim.zero_grad()

    optim.step(ob + ob2 + ob3, lp1, lp2)
    ed_time = time.time()
    # writer.add_scalar('Objective/gradfg', ob.detach(), t_eps)
    # writer.add_scalar('Objective/gradf', lp1.detach(), t_eps)
    # writer.add_scalar('Objective/gradg', lp2.detach(), t_eps)

    if t_eps%100==0:
        print(t_eps)
        torch.save(p1.state_dict(),
                   '../' + folder_location + experiment_name + 'model/agent1_' + str(
                       t_eps) + ".pth")
        torch.save(p2.state_dict(),
                   '../' + folder_location + experiment_name + 'model/agent2_' + str(
                       t_eps) + ".pth")