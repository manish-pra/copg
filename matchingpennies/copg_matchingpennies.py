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
experiment_name = 'copg/'
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

# Initialisation of CPG and game environement
optim = CoPG(p1.parameters(),p2.parameters(), lr =0.5)
env = pennies_game()

num_episode = 150
batch_size = 1000

## Competitive Policy gradient
p1 = policy1()
p2 = policy2()
# Initialisation of CPG and game environement
optim = CoPG(p1.parameters(), p2.parameters(), lr=0.5)
env = pennies_game()

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

    pi_a2_s = p2()
    dist_pi2 = Categorical(pi_a2_s)
    log_probs2 = dist_pi2.log_prob(action_both[:, 1])

    objective = log_probs1 * log_probs2 * (val1_p)
    if objective.size(0) != 1:
        raise 'error'

    ob = objective.mean()

    s_log_probs1 = log_probs1.clone()  # otherwise it doesn't change values
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

    optim.step(ob, lp1, lp2)

    if t_eps%100==0:
        print(t_eps)
        torch.save(p1.state_dict(),
                   '../' + folder_location + experiment_name + 'model/agent1_' + str(
                       t_eps) + ".pth")
        torch.save(p2.state_dict(),
                   '../' + folder_location + experiment_name + 'model/agent2_' + str(
                       t_eps) + ".pth")
