# Game imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, '..')
from open_spiel.python import rl_environment

import torch
from torch.distributions import Categorical
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from markov_soccer.networks import policy
from markov_soccer.networks import critic

from copg_optim.critic_functions import critic_update, get_advantage
from markov_soccer.soccer_state import get_relative_state, get_two_state

folder_location = 'tensorboard/soccer/'
experiment_name = 'gda/'
directory = '../' + folder_location + experiment_name + 'model'

if not os.path.exists(directory):
    os.makedirs(directory)

writer = SummaryWriter('../' + folder_location + experiment_name + 'data')
p1 = policy(12,5)
p2 = policy(12,5)
q = critic(12)
optim_q = torch.optim.Adam(q.parameters(), lr=0.001)
optim_p1 = torch.optim.SGD(p1.parameters(), lr=0.01) # 0.01 also doesn't work
optim_p2 = torch.optim.SGD(p2.parameters(), lr=0.01)


num_episode = 10000
batch_size = 10

game = "markov_soccer"
num_players = 2

env = rl_environment.Environment(game)
num_actions = env.action_spec()["num_actions"]
print(num_actions)


for t_eps in range(num_episode):
    mat_action = []

    mat_state1 = []
    mat_reward1 = []
    mat_done = []

    mat_state2 = []
    mat_reward2 = []
    mat_log_probs1 = []
    mat_log_probs2 = []
    time_step = env.reset()
    a_status, b_status, rel_state = get_relative_state(time_step)
    a_status, b_status, rel_state1, rel_state2 = get_two_state(time_step)
    pre_a = a_status
    pre_b = b_status

    #data_collection
    i = 0
    while time_step.last()==False:
        i=i+1
        action1_prob = p1(torch.FloatTensor(rel_state1))
        #print('a1',action1_prob)
        dist1 = Categorical(action1_prob)
        action1 = dist1.sample() # it will learn probablity for actions and output 0,1,2,3 as possibility of actions
        mat_log_probs1.append(dist1.log_prob(action1).detach())

        # print(action1)

        action2_prob = p2(torch.FloatTensor(rel_state2))
        #print('a2',action2_prob)
        dist2 = Categorical(action2_prob)
        action2 = dist2.sample()
        mat_log_probs2.append(dist2.log_prob(action2).detach())

        action = np.array([action1, action2])

        mat_state1.append(torch.FloatTensor(rel_state1))
        mat_state2.append(torch.FloatTensor(rel_state2))
        mat_action.append(torch.FloatTensor(action))

        time_step = env.step(action)
        a_status, b_status, rel_state = get_relative_state(time_step)
        a_status, b_status, rel_state1, rel_state2 = get_two_state(time_step)
        #print(pretty_board(time_step))
        reward1 = time_step.rewards[0]
        reward2 = time_step.rewards[1]

        reward1 = time_step.rewards[0]
        reward2 = time_step.rewards[1]

        mat_reward1.append(torch.FloatTensor([reward1]))
        mat_reward2.append(torch.FloatTensor([reward2]))

        if time_step.last() == True:
            done = True
        else:
            done=False

        mat_done.append(torch.FloatTensor([1 - done]))
        if done == True:
            time_step = env.reset()
            a_status, b_status, rel_state1, rel_state2 = get_two_state(time_step)
            break
    print(t_eps, i, "done reached")
    if reward1 > 0:
        print("a won")
    if reward2 > 0:
        print("b won")

    # st_q_time = time.time()
    if reward1>0:
        writer.add_scalar('Steps/agent1', i, t_eps)
        writer.add_scalar('Steps/agent2', 50, t_eps)
    elif reward2>0:
        writer.add_scalar('Steps/agent1', 50, t_eps)
        writer.add_scalar('Steps/agent2', i, t_eps)
    else:
        writer.add_scalar('Steps/agent1', 50, t_eps)
        writer.add_scalar('Steps/agent2', 50, t_eps)

    writer.add_scalar('Entropy/agent1', dist1.entropy(), t_eps)
    writer.add_scalar('Entropy/agent2', dist2.entropy(), t_eps)

    val1 = q(torch.stack(mat_state1))
    val1 = val1.detach()
    next_value = 0  # because currently we end ony when its done which is equivalent to no next state
    returns_np1 = get_advantage(next_value, torch.stack(mat_reward1), val1, torch.stack(mat_done), gamma=0.99, tau=0.95)

    returns1 = torch.cat(returns_np1)
    advantage_mat1 = returns1 - val1.transpose(0,1)

    for loss_critic, gradient_norm in critic_update(torch.stack(mat_state1), returns1.view(-1,1), q, optim_q):
        writer.add_scalar('Loss/critic', loss_critic, t_eps)
        #print('critic_update')

    pi_a1_s = p1(torch.stack(mat_state1))
    dist_batch = Categorical(pi_a1_s)
    action_both = torch.stack(mat_action)
    log_probs1 = dist_batch.log_prob(action_both[:, 0])

    val1_p = advantage_mat1
    objective = -log_probs1 * (val1_p)
    ob = objective.mean()
    optim_p1.zero_grad()
    ob.backward()
    optim_p1.step()

    val2_p = -advantage_mat1
    pi_a2_s = p2(torch.stack(mat_state2))
    dist_batch = Categorical(pi_a2_s)
    log_probs2 = dist_batch.log_prob(action_both[:, 1])

    objective2 = -log_probs2 * (val2_p)
    optim_p2.zero_grad()
    ob2 = objective2.mean()
    ob2.backward()
    optim_p2.step()

    if t_eps%100==0:
        torch.save(p1.state_dict(),
                   '../' + folder_location + experiment_name + 'model/agent1_' + str(
                       t_eps) + ".pth")
        torch.save(p2.state_dict(),
                   '../' + folder_location + experiment_name + 'model/agent2_' + str(
                       t_eps) + ".pth")