import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, '..')
from copg_optim import CoPG
import numpy as np
from LQ_game.lq_game import LQ_game
from torch.utils.tensorboard import SummaryWriter

from LQ_game.network import policy1, policy2
from LQ_game.network import critic

from copg_optim.critic_functions import critic_update, get_advantage
import time

import sys
import os


folder_location = 'tensorboard/lq_game/'#'tensorboard/lq_game2/15_04_b1000/'
experiment_name = 'cpg_lr0_1/'
directory = '../' + folder_location + experiment_name + 'model'

if not os.path.exists(directory):
    os.makedirs(directory)

writer = SummaryWriter('../' + folder_location + experiment_name + 'data')
p1 = policy1(1,1)
p2 = policy2(1,1)

for p in p1.parameters():
    print(p)
optim = CoPG(p1.parameters(),p2.parameters(),lr = 0.01) #0.0001


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
            action1 = dist1.sample() # it will learn probablity for actions and output 0,1,2,3 as possibility of actions

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
            # state = torch.FloatTensor(state).reshape(1)
            #print(state,action1,action2,reward1)
            # print('a1', action_app[0], reward1, action1_prob)
            # print('a2', action_app[1], reward2, action2_prob)
            #print(state, reward, done)
            mat_reward1.append(torch.FloatTensor([reward1]))
            mat_reward2.append(torch.FloatTensor([reward2]))
            if i==horizon-1:
                done = True
            mat_done.append(torch.FloatTensor([1 - done]))
        writer.add_scalar('Reward/zerosum', torch.mean(torch.stack(mat_reward1)), t_eps)

    writer.add_scalar('Mean/Agent1', dist1.mean, t_eps)
    writer.add_scalar('Mean/agent2', dist2.mean, t_eps)
    mat_action1_tensor = torch.stack(mat_action1)
    mat_action2_tensor = torch.stack(mat_action2)
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

    val1_p = advantage_mat1.transpose(0,1)#torch.stack(mat_reward1)

    if val1_p.size(1)!=1:
        raise 'error'

    pi_a1_s = p1(torch.stack(mat_state1))
    log_probs1 = pi_a1_s.log_prob(mat_action1_tensor)

    pi_a2_s = p2(torch.stack(mat_state2))
    log_probs2 = pi_a2_s.log_prob(mat_action2_tensor)

    objective = log_probs1*log_probs2*(val1_p)
    if objective.size(1)!=1:
        raise 'error'

    ob = objective.mean()

    s_log_probs1 = log_probs1[0:log_probs1.size(0)].clone() # otherwise it doesn't change values
    s_log_probs2 = log_probs2[0:log_probs2.size(0)].clone()

    #if horizon is 4, log_probs1.size =4 and for loop will go till [0,3]

    mask = torch.stack(mat_done)

    s_log_probs1[0] = 0
    s_log_probs2[0] = 0

    for i in range(1,log_probs1.size(0)):
        s_log_probs1[i] = torch.add(s_log_probs1[i - 1], log_probs1[i-1])*mask[i-1]
        s_log_probs2[i] = torch.add(s_log_probs2[i - 1], log_probs2[i-1])*mask[i-1]

    objective2 = s_log_probs1[1:s_log_probs1.size(0)]*log_probs2[1:log_probs2.size(0)]*(val1_p[1:val1_p.size(0)])
    ob2 = objective2.sum()/((horizon-1)*batch_size)

    objective3 = log_probs1[1:log_probs1.size(0)]*s_log_probs2[1:s_log_probs2.size(0)]*(val1_p[1:val1_p.size(0)])
    ob3 = objective3.sum()/((horizon-1)*batch_size)

    # for j in p1.parameters():
    #     print(j)
    #print(advantage_mat1.mean(),advantage_mat2.mean())
    lp1 = log_probs1*val1_p
    lp1=lp1.mean()
    lp2 = log_probs2*val1_p
    lp2=lp2.mean()
    optim.zero_grad()
    print(ob)
    writer.add_scalar('Objective/gradfg', ob.detach(), t_eps)
    writer.add_scalar('Objective/gradf', lp1.detach(), t_eps)
    writer.add_scalar('Objective/gradg', lp2.detach(), t_eps)
    optim.step(ob + ob2 + ob3, lp1, lp2)

    norm_gx, norm_gy, norm_px, norm_py, norm_cgx, norm_cgy, _, _= optim.getinfo()
    writer.add_scalar('grad/norm_gx', norm_gx, t_eps)
    writer.add_scalar('grad/norm_gy', norm_gy, t_eps)
    writer.add_scalar('grad/norm_px', norm_px, t_eps)
    writer.add_scalar('grad/norm_py', norm_py, t_eps)
    writer.add_scalar('grad/norm_cgx', norm_cgx, t_eps)
    writer.add_scalar('grad/norm_cgy', norm_cgy, t_eps)

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