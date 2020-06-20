# Game imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, '..')
from copg_optim import RCoPG as CoPG
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from car_racing.network import Actor#Var as Actor
from car_racing.network import Critic

from copg_optim.critic_functions import critic_update, get_advantage
import time

import json
import sys

import car_racing_simulator.VehicleModel as VehicleModel
import car_racing_simulator.Track as Track
from car_racing.orca_env_function import getreward, getdone, getfreezereward, getfreezecollosionreward, getfreezecollosionReachedreward, getfreezeTimecollosionReachedreward
import random

folder_location = 'tensorboard/orca/'
experiment_name = 'gda/'

directory = '../' + folder_location + experiment_name + 'model'
if not os.path.exists(directory):
    os.makedirs(directory)
writer = SummaryWriter('../' + folder_location + experiment_name + 'data')
config = json.load(open('config.json'))

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = 'cpu' #torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


vehicle_model = VehicleModel.VehicleModel(config["n_batch"], device, config)

x0 = torch.zeros(config["n_batch"], config["n_state"])
# x0[:, 3] = 0

u0 = torch.zeros(config["n_batch"], config["n_control"])

p1 = Actor(10,2, std=0.1)
p2 = Actor(10,2, std=0.1)
q = Critic(10)


optim_p1 = torch.optim.SGD(p1.parameters(), 3e-5) # 0.00008
optim_p2 = torch.optim.SGD(p2.parameters(), 3e-5) # 0.00008
optim_q = torch.optim.Adam(q.parameters(), 0.008)

batch_size = 8
num_episode = 20000

for t_eps in range(num_episode):
    mat_action1 = []
    mat_action2 = []

    mat_state1 = []
    mat_reward1 = []
    mat_done = []

    mat_state2 = []
    mat_reward2 = []
    print(t_eps)

    #data_collection
    avg_itr = 0

    curr_batch_size = 2

    #state = torch.zeros(config["n_batch"], config["n_state"])
    state_c1 = torch.zeros(curr_batch_size, config["n_state"])#state[:,0:6].view(6)
    state_c2 = torch.zeros(curr_batch_size, config["n_state"])#state[:, 6:12].view(6)
    init_p1 = torch.zeros((curr_batch_size)) #5*torch.rand((curr_batch_size))
    init_p2 = torch.zeros((curr_batch_size)) #5*torch.rand((curr_batch_size))
    state_c1[:,0] = init_p1
    state_c2[:,0] = init_p2
    a = random.choice([-0.1,0.1])
    b = a*(-1)
    state_c1[:, 1] = a*torch.ones((curr_batch_size))
    state_c2[:, 1] = b*torch.ones((curr_batch_size))
    batch_mat_state1 = torch.empty(0)
    batch_mat_state2 = torch.empty(0)
    batch_mat_action1 = torch.empty(0)
    batch_mat_action2 = torch.empty(0)
    batch_mat_reward1 = torch.empty(0)
    batch_mat_done = torch.empty(0)

    itr = 0
    done = torch.tensor([False])
    done_c1 = torch.zeros((curr_batch_size)) <= -0.1
    done_c2 = torch.zeros((curr_batch_size)) <= -0.1
    prev_coll_c1 = torch.zeros((curr_batch_size)) <= -0.1
    prev_coll_c2 = torch.zeros((curr_batch_size)) <= -0.1
    counter1 = torch.zeros((curr_batch_size))
    counter2 = torch.zeros((curr_batch_size))

    #for itr in range(50):
    while np.all(done.numpy()) == False:
        avg_itr+=1

        st1_gpu = torch.cat([state_c1[:,0:5],state_c2[:,0:5]],dim=1).to(device)

        dist1 = p1(st1_gpu)
        action1 = dist1.sample().to('cpu')

        st2_gpu = torch.cat([state_c2[:, 0:5], state_c1[:, 0:5]], dim=1).to(device)

        dist2 = p2(st2_gpu)
        action2 = dist2.sample().to('cpu')

        if itr>0:
            mat_state1 = torch.cat([mat_state1.view(-1,curr_batch_size,5),state_c1[:,0:5].view(-1,curr_batch_size,5)],dim=0) # concate along dim = 0
            mat_state2 = torch.cat([mat_state2.view(-1, curr_batch_size, 5), state_c2[:, 0:5].view(-1, curr_batch_size, 5)], dim=0)
            mat_action1 = torch.cat([mat_action1.view(-1, curr_batch_size, 2), action1.view(-1, curr_batch_size, 2)], dim=0)
            mat_action2 = torch.cat([mat_action2.view(-1, curr_batch_size, 2), action2.view(-1, curr_batch_size, 2)], dim=0)
        else:
            mat_state1 = state_c1[:,0:5]
            mat_state2 = state_c2[:, 0:5]
            mat_action1 = action1
            mat_action2 = action2
        #mat_state2.append(state_c2[:,0:5])
        # mat_action1.append(action1)
        # mat_action2.append(action2)

        prev_state_c1 = state_c1
        prev_state_c2 = state_c2

        state_c1 = vehicle_model.dynModelBlendBatch(state_c1.view(-1,6), action1.view(-1,2)).view(-1,6)
        state_c2 = vehicle_model.dynModelBlendBatch(state_c2.view(-1,6), action2.view(-1,2)).view(-1,6)

        state_c1 = (state_c1.transpose(0, 1) * (~done_c1) + prev_state_c1.transpose(0, 1) * (done_c1)).transpose(0, 1)
        state_c2 = (state_c2.transpose(0, 1) * (~done_c2) + prev_state_c2.transpose(0, 1) * (done_c2)).transpose(0, 1)

        reward1, reward2, done_c1, done_c2, coll_c1, coll_c2, counter1, counter2 = getfreezeTimecollosionReachedreward(state_c1, state_c2,
                                                                     vehicle_model.getLocalBounds(state_c1[:, 0]),
                                                                     vehicle_model.getLocalBounds(state_c2[:, 0]),
                                                                     prev_state_c1, prev_state_c2, prev_coll_c1, prev_coll_c2, counter1, counter2)

        # reward1, reward2, done_c1, done_c2, coll_c1, coll_c2 = getfreezecollosionReachedreward(state_c1, state_c2,
        #                                                              vehicle_model.getLocalBounds(state_c1[:, 0]),
        #                                                              vehicle_model.getLocalBounds(state_c2[:, 0]),
        #                                                              prev_state_c1, prev_state_c2, prev_coll_c1, prev_coll_c2)

        done = (done_c1) * (done_c2)  # ~((~done_c1) * (~done_c2))
        # done =  ~((~done_c1) * (~done_c2))
        mask_ele = ~done

        if itr>0:
            mat_reward1 = torch.cat([mat_reward1.view(-1,curr_batch_size,1),reward1.view(-1,curr_batch_size,1)],dim=0) # concate along dim = 0
            mat_done = torch.cat([mat_done.view(-1, curr_batch_size, 1), mask_ele.view(-1, curr_batch_size, 1)], dim=0)
        else:
            mat_reward1 = reward1
            mat_done = mask_ele

        remaining_xo = ~done

        state_c1 = state_c1[remaining_xo]
        state_c2 = state_c2[remaining_xo]
        prev_coll_c1 = coll_c1[remaining_xo]#removing elements that died
        prev_coll_c2 = coll_c2[remaining_xo]#removing elements that died
        counter1 = counter1[remaining_xo]
        counter2 = counter2[remaining_xo]

        curr_batch_size = state_c1.size(0)

        if curr_batch_size<remaining_xo.size(0):
            if batch_mat_action1.nelement() == 0:
                batch_mat_state1 = mat_state1.transpose(0, 1)[~remaining_xo].view(-1, 5)
                batch_mat_state2 = mat_state2.transpose(0, 1)[~remaining_xo].view(-1, 5)
                batch_mat_action1 = mat_action1.transpose(0, 1)[~remaining_xo].view(-1, 2)
                batch_mat_action2 = mat_action2.transpose(0, 1)[~remaining_xo].view(-1, 2)
                batch_mat_reward1 = mat_reward1.transpose(0, 1)[~remaining_xo].view(-1, 1)
                batch_mat_done = mat_done.transpose(0, 1)[~remaining_xo].view(-1, 1)
                # progress_done1 = batch_mat_state1[batch_mat_state1.size(0) - 1, 0] - batch_mat_state1[0, 0]
                # progress_done2 = batch_mat_state2[batch_mat_state2.size(0) - 1, 0] - batch_mat_state2[0, 0]
                progress_done1 = torch.sum(mat_state1.transpose(0, 1)[~remaining_xo][:,mat_state1.size(0)-1,0] - mat_state1.transpose(0, 1)[~remaining_xo][:,0,0])
                progress_done2 = torch.sum(mat_state2.transpose(0, 1)[~remaining_xo][:,mat_state2.size(0)-1,0] - mat_state2.transpose(0, 1)[~remaining_xo][:,0,0])
                element_deducted = ~(done_c1*done_c2)
                done_c1 = done_c1[element_deducted]
                done_c2 = done_c2[element_deducted]
            else:
                prev_size = batch_mat_state1.size(0)
                batch_mat_state1 = torch.cat([batch_mat_state1,mat_state1.transpose(0, 1)[~remaining_xo].view(-1,5)],dim=0)
                batch_mat_state2 = torch.cat([batch_mat_state2, mat_state2.transpose(0, 1)[~remaining_xo].view(-1, 5)],dim=0)
                batch_mat_action1 = torch.cat([batch_mat_action1, mat_action1.transpose(0, 1)[~remaining_xo].view(-1, 2)],dim=0)
                batch_mat_action2 = torch.cat([batch_mat_action2, mat_action2.transpose(0, 1)[~remaining_xo].view(-1, 2)],dim=0)
                batch_mat_reward1 = torch.cat([batch_mat_reward1, mat_reward1.transpose(0, 1)[~remaining_xo].view(-1, 1)],dim=0)
                batch_mat_done = torch.cat([batch_mat_done, mat_done.transpose(0, 1)[~remaining_xo].view(-1, 1)],dim=0)
                progress_done1 = progress_done1 + torch.sum(mat_state1.transpose(0, 1)[~remaining_xo][:, mat_state1.size(0) - 1, 0] -
                                           mat_state1.transpose(0, 1)[~remaining_xo][:, 0, 0])
                progress_done2 = progress_done2 + torch.sum(mat_state2.transpose(0, 1)[~remaining_xo][:, mat_state2.size(0) - 1, 0] -
                                           mat_state2.transpose(0, 1)[~remaining_xo][:, 0, 0])
                # progress_done1 = progress_done1 + batch_mat_state1[batch_mat_state1.size(0) - 1, 0] - batch_mat_state1[prev_size, 0]
                # progress_done2 = progress_done2 + batch_mat_state2[batch_mat_state2.size(0) - 1, 0] - batch_mat_state2[prev_size, 0]
                element_deducted = ~(done_c1*done_c2)
                done_c1 = done_c1[element_deducted]
                done_c2 = done_c2[element_deducted]

            mat_state1 = mat_state1.transpose(0, 1)[remaining_xo].transpose(0, 1)
            mat_state2 = mat_state2.transpose(0, 1)[remaining_xo].transpose(0, 1)
            mat_action1 = mat_action1.transpose(0, 1)[remaining_xo].transpose(0, 1)
            mat_action2 = mat_action2.transpose(0, 1)[remaining_xo].transpose(0, 1)
            mat_reward1 = mat_reward1.transpose(0, 1)[remaining_xo].transpose(0, 1)
            mat_done = mat_done.transpose(0, 1)[remaining_xo].transpose(0, 1)

        # print(avg_itr,remaining_xo.size(0))

        # writer.add_scalar('Reward/agent1', reward1, t_eps)
        itr = itr + 1

        if np.all(done.numpy()) == True or batch_mat_state1.size(0)>900 or itr>400:# or itr>900: #brak only if all elements in the array are true
            prev_size = batch_mat_state1.size(0)
            batch_mat_state1 = torch.cat([batch_mat_state1, mat_state1.transpose(0, 1).reshape(-1, 5)],dim=0)
            batch_mat_state2 = torch.cat([batch_mat_state2, mat_state2.transpose(0, 1).reshape(-1, 5)],dim=0)
            batch_mat_action1 = torch.cat([batch_mat_action1, mat_action1.transpose(0, 1).reshape(-1, 2)],dim=0)
            batch_mat_action2 = torch.cat([batch_mat_action2, mat_action2.transpose(0, 1).reshape(-1, 2)],dim=0)
            batch_mat_reward1 = torch.cat([batch_mat_reward1, mat_reward1.transpose(0, 1).reshape(-1, 1)],dim=0) #should i create a false or true array?
            print("done", itr)
            print(mat_done.shape)
            mat_done[mat_done.size(0)-1,:,:] = torch.ones((mat_done[mat_done.size(0)-1,:,:].shape))>=2 # creating a true array of that shape
            print(mat_done.shape, batch_mat_done.shape)
            if batch_mat_done.nelement() == 0:
                batch_mat_done = mat_done.transpose(0, 1).reshape(-1, 1)
                progress_done1 = 0
                progress_done2 =0
            else:
                batch_mat_done = torch.cat([batch_mat_done, mat_done.transpose(0, 1).reshape(-1, 1)], dim=0)
            if prev_size == batch_mat_state1.size(0):
                progress_done1 = progress_done1
                progress_done2 = progress_done2
            else:
                progress_done1 = progress_done1 + torch.sum(mat_state1.transpose(0, 1)[:, mat_state1.size(0) - 1, 0] -
                                           mat_state1.transpose(0, 1)[:, 0, 0])
                progress_done2 = progress_done2 + torch.sum(mat_state2.transpose(0, 1)[:, mat_state2.size(0) - 1, 0] -
                                           mat_state2.transpose(0, 1)[:, 0, 0])
            print(batch_mat_done.shape)
            # print("done", itr)
            break

    # print(avg_itr)

    print(batch_mat_state1.shape,itr)
    writer.add_scalar('Dist/variance_throttle_p1', dist1.variance[0,0], t_eps)
    writer.add_scalar('Dist/variance_steer_p1', dist1.variance[0,1], t_eps)
    writer.add_scalar('Dist/variance_throttle_p2', dist2.variance[0,0], t_eps)
    writer.add_scalar('Dist/variance_steer_p2', dist2.variance[0,1], t_eps)
    writer.add_scalar('Reward/mean', batch_mat_reward1.mean(), t_eps)
    writer.add_scalar('Reward/sum', batch_mat_reward1.sum(), t_eps)
    writer.add_scalar('Progress/final_p1', progress_done1/batch_size, t_eps)
    writer.add_scalar('Progress/final_p2', progress_done2/batch_size, t_eps)
    writer.add_scalar('Progress/trajectory_length', itr, t_eps)
    writer.add_scalar('Progress/agent1', batch_mat_state1[:,0].mean(), t_eps)
    writer.add_scalar('Progress/agent2', batch_mat_state2[:,0].mean(), t_eps)

    val1 = q(torch.cat([batch_mat_state1,batch_mat_state2],dim=1).to(device))
    val1 = val1.detach().to('cpu')
    next_value = 0  # because currently we end ony when its done which is equivalent to no next state
    returns_np1 = get_advantage(next_value, batch_mat_reward1, val1, batch_mat_done, gamma=0.99, tau=0.95)

    returns1 = torch.cat(returns_np1)
    advantage_mat1 = returns1.view(1,-1) - val1.transpose(0,1)

    # val2 = q(torch.stack(mat_state2))
    # val2 = val2.detach()
    # next_value = 0  # because currently we end ony when its done which is equivalent to no next state
    # returns_np2 = get_advantage(next_value, torch.stack(mat_reward2), val2, torch.stack(mat_done), gamma=0.99, tau=0.95)
    #
    # returns2 = torch.cat(returns_np2)
    # advantage_mat2 = returns2 - val2.transpose(0,1)
    state_gpu_p1 = torch.cat([batch_mat_state1, batch_mat_state2], dim=1).to(device)
    state_gpu_p2 = torch.cat([batch_mat_state2, batch_mat_state1], dim=1).to(device)
    returns1_gpu = returns1.view(-1, 1).to(device)

    for loss_critic, gradient_norm in critic_update(state_gpu_p1,returns1_gpu, q, optim_q):
        writer.add_scalar('Loss/critic', loss_critic, t_eps)
        #print('critic_update')
    ed_q_time = time.time()
    # print('q_time',ed_q_time-st_q_time)
    # print('q_time',ed_q_time-st_q_time)

    #val1_p = -advantage_mat1#val1.detach()
    val1_p = advantage_mat1.to(device)
    writer.add_scalar('Advantage/agent1', advantage_mat1.mean(), t_eps)
    # st_time = time.time()
    # calculate gradients
    dist_batch1 = p1(state_gpu_p1)
    log_probs1 = dist_batch1.log_prob(batch_mat_action1)
    # log_probs1 = log_probs1_inid.sum(1)

    objective = -log_probs1 * val1_p.transpose(0,1) # ob.backward will maximise this
    ob1 = objective.mean()
    optim_p1.zero_grad()
    ob1.backward()
    total_norm = 0
    # for p in p1.parameters():
    #     param_norm = p.grad.data.norm(2)
    #     total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    optim_p1.step()
    writer.add_scalar('Objective/gradnorm1', total_norm, t_eps)

    dist_batch2 = p2(state_gpu_p2)
    log_probs2 = dist_batch2.log_prob(batch_mat_action2)
    # log_probs2 = log_probs2_inid.sum(1)
    objective = log_probs2 * val1_p.transpose(0,1) # ob.backward will minimise this
    ob2 = objective.mean()
    optim_p2.zero_grad()
    ob2.backward()
    total_norm = 0
    # for p in p2.parameters():
    #     param_norm = p.grad.data.norm(2)
    #     total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    optim_p2.step()
    writer.add_scalar('Objective/gradnorm2', total_norm, t_eps)

    writer.add_scalar('Entropy/agent1', dist_batch1.entropy().mean().detach(), t_eps)
    writer.add_scalar('Entropy/agent2', dist_batch2.entropy().mean().detach(), t_eps)
    writer.add_scalar('Objective/gradp1', ob1.detach(), t_eps)
    writer.add_scalar('Objective/gradp2', ob2.detach(), t_eps)

    # print('p_time',ed_time-st_time)
        # for j in p1.parameters():
        #     print
    if t_eps%20==0:
            torch.save(p1.state_dict(),
                       '../' + folder_location + experiment_name + 'model/agent1_' + str(
                           t_eps) + ".pth")
            torch.save(p2.state_dict(),
                       '../' + folder_location + experiment_name + 'model/agent2_' + str(
                           t_eps) + ".pth")
            torch.save(q.state_dict(),
                       '../' + folder_location + experiment_name + 'model/val_' + str(
                           t_eps) + ".pth")