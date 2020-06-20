import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, '..')
from LQ_game.lq_game import LQ_game
from torch.utils.tensorboard import SummaryWriter

from LQ_game.network import policy1, policy2
from LQ_game.network import critic

from copg_optim.critic_functions import critic_update, get_advantage
import time

import sys
import os

folder_location = 'tensorboard/lq_game/'
experiment_name = 'gda/'
directory = '../' + folder_location + experiment_name + 'model'

if not os.path.exists(directory):
    os.makedirs(directory)

writer = SummaryWriter('../' + folder_location + experiment_name + 'data')
p1 = policy1(1,1)
p2 = policy2(1,1)

optim_p1 = torch.optim.SGD(p1.parameters(), lr=0.01)  #0.001
optim_p2 = torch.optim.SGD(p2.parameters(), lr=0.01)

for p in p1.parameters():
    print(p)
q = critic(1)
optim_q = torch.optim.Adam(q.parameters(), lr=0.01)

batch_size = 1000
num_episode = 10000
horizon = 5

env = LQ_game()

for t_eps in range(num_episode):
    mat_action1 = []
    mat_action2 = []

    mat_state1 = []
    mat_reward1 = []
    mat_done = []

    mat_state2 = []
    mat_reward2 = []
    #data_collection
    for _ in range(batch_size):
        state, _, _, _, _ = env.reset()
        state = torch.FloatTensor(state).reshape(1)
        for i in range(horizon):
            dist1 = p1(state)
            action1 = dist1.sample()  # it will learn probablity for actions and output 0,1,2,3 as possibility of actions

            dist2 = p2(state)
            action2 = dist2.sample()

            mat_state1.append(torch.FloatTensor(state))
            mat_state2.append(torch.FloatTensor(state))
            mat_action1.append(torch.FloatTensor(action1))
            mat_action2.append(torch.FloatTensor(action2))
            #print(action)
            writer.add_scalar('State/State', state, i)
            state, reward1, reward2, done, _ = env.step(action1,action2)
            state = state.reshape(1)
            #print(mat_state1)
            # print('a1', action_app[0], reward1, action1_prob)
            # print('a2', action_app[1], reward2, action2_prob)
            #print(state, reward, done)
            mat_reward1.append(torch.FloatTensor([reward1]))
            mat_reward2.append(torch.FloatTensor([reward2]))
            if i==horizon-1:
                done = True
            mat_done.append(torch.FloatTensor([1 - done]))
        writer.add_scalar('Reward/zerosum', torch.mean(torch.stack(mat_reward1)), t_eps)

    #print(action)
    # print('a1',dist1.mean, dist1.variance)
    # print('a2',dist2.mean, dist2.variance)
    writer.add_scalar('Mean/Agent1', dist1.mean, t_eps)
    writer.add_scalar('Mean/agent2', dist2.mean, t_eps)
    mat_action1_tensor = torch.stack(mat_action1)
    mat_action2_tensor = torch.stack(mat_action2)
    print(dist1.mean)
    # writer.add_scalar('Variance/Agent1', dist1.variance, t_eps)
    # writer.add_scalar('Variance/agent2', dist2.variance, t_eps)

    writer.add_scalar('Action/Agent1', torch.mean(mat_action1_tensor), t_eps)
    writer.add_scalar('Action/agent2', torch.mean(mat_action2_tensor), t_eps)

    for l,p in enumerate(p1.parameters()):
        if l==0:
            writer.add_scalar('Controller/Agent1', p.data, t_eps)
        else:
            writer.add_scalar('controller_std/Agent1', p.data, t_eps)


    for l,p in enumerate(p2.parameters()):
        if l==0:
            writer.add_scalar('Controller/Agent2', p.data, t_eps)
        else:
            writer.add_scalar('controller_std/Agent2', p.data, t_eps)


    # writer.add_scalar('Action_prob/agent1', action1_prob[3], t_eps)
    # writer.add_scalar('Action_prob/agent2', action2_prob[3], t_eps)

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

    writer.add_scalar('Advantage/agent1', advantage_mat1.mean(), t_eps)
    writer.add_scalar('Advantage/agent2', advantage_mat2.mean(), t_eps)

    #for loss_critic, gradient_norm in critic_update(torch.cat([torch.stack(mat_state1),torch.stack(mat_state2)]), torch.cat([returns1,returns2]).view(-1,1), num_ppo_epochs, size_mini_batch, q, optim_q):
    for loss_critic, gradient_norm in critic_update(torch.stack(mat_state1), returns1.view(-1, 1), q, optim_q):
        writer.add_scalar('Loss/critic', loss_critic, t_eps)
        #print('critic_update')

    ed_q_time = time.time()
    # print('q_time',ed_q_time-st_q_time)

    val1_p = advantage_mat1.transpose(0,1)#torch.stack(mat_reward1)
    # st_time = time.time()
    # calculate gradients
    pi_a1_s = p1(torch.stack(mat_state1))
    log_probs1 = pi_a1_s.log_prob(mat_action1_tensor)
    objective = -log_probs1 * (val1_p)
    ob = objective.mean()
    optim_p1.zero_grad()
    ob.backward()
    total_norm = 0
    for p in p1.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    optim_p1.step()
    writer.add_scalar('grad/p1', total_norm, t_eps)

    val2_p = -val1_p
    pi_a2_s = p2(torch.stack(mat_state2))
    log_probs2 = pi_a2_s.log_prob(mat_action2_tensor)

    objective2 = -log_probs2 * (val2_p)
    optim_p2.zero_grad()
    ob2 = objective2.mean()
    ob2.backward()
    total_norm = 0
    for p in p2.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    optim_p2.step()
    writer.add_scalar('Objective/agent2', ob2, t_eps)
    writer.add_scalar('grad/p2', torch.mean(torch.stack(mat_reward1)), t_eps)
    # print('p_time',ed_time-st_time)
        # for j in p1.parameters():
        #     print


    if t_eps%100==0:
        print(t_eps)
        torch.save(p1.state_dict(),
                   '../' + folder_location + experiment_name + 'model/agent1_' + str(
                       t_eps) + ".pth")
        torch.save(p2.state_dict(),
                   '../' + folder_location + experiment_name + 'model/agent2_' + str(
                       t_eps) + ".pth")
        torch.save(q.state_dict(),
                   '../' + folder_location + experiment_name + 'model/val_' + str(
                       t_eps) + ".pth")