# Game imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, '..')
from open_spiel.python import rl_environment
import torch
from copg_optim import CoPG
from torch.distributions import Categorical
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from markov_soccer.networks import policy
from markov_soccer.networks import critic

from copg_optim.critic_functions import critic_update, get_advantage
from markov_soccer.soccer_state import get_relative_state, get_two_state
import time

folder_location = 'tensorboard/machingpennies/'
experiment_name = 'copg/'
directory = '../' + folder_location + experiment_name + 'model'

if not os.path.exists(directory):
    os.makedirs(directory)
writer = SummaryWriter('../' + folder_location + experiment_name + 'data')

p1 = policy(12,5)
q = critic(12)

optim_q = torch.optim.Adam(q.parameters(), lr=0.001)

optim = CoPG(p1.parameters(),p1.parameters(), lr=1e-2)

batch_size = 10
num_episode = 30000

game = "markov_soccer"
num_players = 2

env = rl_environment.Environment(game)
num_actions = env.action_spec()["num_actions"]

for t_eps in range(num_episode):
    mat_action = []

    mat_state1 = []
    mat_reward1 = []
    mat_done = []

    mat_state2 = []
    mat_reward2 = []
    time_step = env.reset()
    # print(pretty_board(time_step))
    a_status, b_status, rel_state1, rel_state2 = get_two_state(time_step)

    #data_collection
    avg_itr = 0
    for _ in range(batch_size):
        i = 0
        while time_step.last()==False:
            action1_prob = p1(torch.FloatTensor(rel_state1))
            #print('a1',action1_prob)
            dist1 = Categorical(action1_prob)
            action1 = dist1.sample() # it will learn probablity for actions and output 0,1,2,3 as possibility of actions
            # print(action1)
            avg_itr = avg_itr + 1

            action2_prob = p1(torch.FloatTensor(rel_state2))
            #print('a2',action2_prob)
            dist2 = Categorical(action2_prob)
            action2 = dist2.sample()

            action = np.array([action1, action2])

            mat_state1.append(torch.FloatTensor(rel_state1))
            mat_state2.append(torch.FloatTensor(rel_state2))
            mat_action.append(torch.FloatTensor(action))

            time_step = env.step(action)
            a_status, b_status, rel_state1, rel_state2 = get_two_state(time_step)

            reward1 = time_step.rewards[0]
            reward2 = time_step.rewards[1]

            #print(pretty_board(time_step))
            mat_reward1.append(torch.FloatTensor([reward1]))
            mat_reward2.append(torch.FloatTensor([reward2]))

            if time_step.last() == True:
                done = True
            else:
                done=False

            mat_done.append(torch.FloatTensor([1 - done]))
            i=i+1
            if done == True:
                time_step = env.reset()
                # print(pretty_board(time_step))
                a_status, b_status, rel_state1, rel_state2 = get_two_state(time_step)
                #print(state1, state2, reward1,reward2, done)
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

    val1 = q(torch.stack(mat_state1))
    val1 = val1.detach()
    next_value = 0  # because currently we end ony when its done which is equivalent to no next state
    returns_np1 = get_advantage(next_value, torch.stack(mat_reward1), val1, torch.stack(mat_done), gamma=0.99, tau=0.95)

    returns1 = torch.cat(returns_np1)
    advantage_mat1 = returns1 - val1.transpose(0,1)

    val2 = q(torch.stack(mat_state2))
    val2 = val2.detach()
    next_value = 0  # because currently we end ony when its done which is equivalent to no next state
    returns_np2 = get_advantage(next_value, torch.stack(mat_reward2), val2, torch.stack(mat_done), gamma=0.99, tau=0.95)

    returns2 = torch.cat(returns_np2)
    advantage_mat2 = returns2 - val2.transpose(0,1)

    for loss_critic, gradient_norm in critic_update(torch.cat([torch.stack(mat_state1),torch.stack(mat_state2)]), torch.cat([returns1,returns2]).view(-1,1), q, optim_q):
        writer.add_scalar('Loss/critic', loss_critic, t_eps)
    ed_q_time = time.time()

    val1_p = advantage_mat1

    pi_a1_s = p1(torch.stack(mat_state1))
    dist_batch1 = Categorical(pi_a1_s)
    action_both = torch.stack(mat_action)
    log_probs1 = dist_batch1.log_prob(action_both[:,0])

    pi_a2_s = p1(torch.stack(mat_state2))
    dist_batch2 = Categorical(pi_a2_s)
    log_probs2 = dist_batch2.log_prob(action_both[:,1])

    objective = log_probs1*log_probs2*(val1_p)
    if objective.size(0)!=1:
        raise 'error'

    ob = objective.mean()

    s_log_probs1 = log_probs1[0:log_probs1.size(0)].clone() # otherwise it doesn't change values
    s_log_probs2 = log_probs2[0:log_probs2.size(0)].clone()


    mask = torch.stack(mat_done)

    s_log_probs1[0] = 0
    s_log_probs2[0] = 0

    for i in range(1,log_probs1.size(0)):
        s_log_probs1[i] = torch.add(s_log_probs1[i - 1], log_probs1[i-1])*mask[i-1]
        s_log_probs2[i] = torch.add(s_log_probs2[i - 1], log_probs2[i-1])*mask[i-1]

    objective2 = s_log_probs1[1:s_log_probs1.size(0)]*log_probs2[1:log_probs2.size(0)]*(val1_p[0,1:val1_p.size(1)])
    ob2 = objective2.sum()/(objective2.size(0)-batch_size+1)


    objective3 = log_probs1[1:log_probs1.size(0)]*s_log_probs2[1:s_log_probs2.size(0)]*(val1_p[0,1:val1_p.size(1)])
    ob3 = objective3.sum()/(objective3.size(0)-batch_size+1)


    lp1 = log_probs1*val1_p
    lp1=lp1.mean()
    lp2 = log_probs2*val1_p
    lp2=lp2.mean()
    optim.zero_grad()
    optim.step(ob + ob2 + ob3, lp1,lp2)
    ed_time = time.time()

    writer.add_scalar('Entropy/agent1', dist_batch1.entropy().mean().detach(), t_eps)
    writer.add_scalar('Entropy/agent2', dist_batch2.entropy().mean().detach(), t_eps)

    if t_eps%100==0:
        torch.save(p1.state_dict(),
                   '../' + folder_location + experiment_name + 'model/agent1_' + str(
                       t_eps) + ".pth")
        torch.save(q.state_dict(),
                   '../' + folder_location + experiment_name + 'model/val_' + str(
                       t_eps) + ".pth")
        # torch.save(p2.state_dict(),
        #            '../' + folder_location + experiment_name + 'model/agent2_' + str(
        #                t_eps) + ".pth")