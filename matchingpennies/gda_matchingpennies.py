import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, '..')
from copg_optim import CoPG
from torch.distributions import Categorical
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matchingpennies.matching_pennies import pennies_game
from matchingpennies.network import policy1, policy2

folder_location = 'tensorboard/machingpennies/'
experiment_name = 'gda/'
directory = '../' + folder_location + experiment_name + 'model'

if not os.path.exists(directory):
    os.makedirs(directory)
writer = SummaryWriter('../' + folder_location + experiment_name + 'data')

# Initialize policy for both agents for rock paper scissors
p1 = policy1()
p2 = policy2()
for p in p1.parameters():
    print(p)
for p in p2.parameters():
    print(p)

optim_p1 = torch.optim.SGD(p1.parameters(), lr=0.5)
optim_p2 = torch.optim.SGD(p2.parameters(), lr=0.5)
env = pennies_game()

num_episode = 150
batch_size = 1000

for t_eps in range(num_episode):
    mat_action = []
    mat_state1 = []
    mat_reward1 = []
    mat_done = []
    mat_state2 = []
    mat_reward2 = []
    state, _, _, _, _ = env.reset()

    # data_collection
    for i in range(batch_size):
        pi1 = p1()
        dist1 = Categorical(pi1)
        action1 = dist1.sample()

        pi2 = p2()
        dist2 = Categorical(pi2)
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
    dist_pi1 = Categorical(pi_a1_s)
    action_both = torch.stack(mat_action)
    writer.add_scalar('Entropy/Agent1', dist1.entropy().data, t_eps)
    writer.add_scalar('Entropy/agent2', dist2.entropy().data, t_eps)

    writer.add_scalar('Action/Agent1', torch.mean(action_both[:,0]), t_eps)
    writer.add_scalar('Action/agent2', torch.mean(action_both[:,1]), t_eps)
    writer.add_scalar('Agent1/sm1', pi1.data[0], t_eps)
    writer.add_scalar('Agent1/sm2', pi1.data[1], t_eps)
    writer.add_scalar('Agent2/Agent1', pi2.data[0], t_eps)
    writer.add_scalar('Agent2/agent2', pi2.data[1], t_eps)

    log_probs1 = dist_pi1.log_prob(action_both[:, 0])

    objective = -log_probs1 * (val1_p)
    ob = objective.mean()
    optim_p1.zero_grad()
    ob.backward()
    optim_p1.step()

    val2_p = -val1_p
    pi_a2_s = p2()
    dist_pi2 = Categorical(pi_a2_s)
    log_probs2 = dist_pi2.log_prob(action_both[:, 1])

    objective2 = -log_probs2 * (val2_p)
    optim_p2.zero_grad()
    ob2 = objective2.mean()
    ob2.backward()
    optim_p2.step()
