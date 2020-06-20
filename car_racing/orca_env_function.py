import torch

import math
import numpy as np

l_f = 0.029
l_r = 0.033

W = 0.03

def getreward(state_c1,state_c2, lb_c1, lb_c2, prev_state_c1, prev_state_c2):
    r1 = state_c1[:, 0] - prev_state_c1[:, 0]
    r2 = state_c2[:, 0] - prev_state_c2[:, 0]
    #reward = state_c1[:,0] - state_c2[:,0]
    reward = 50*(r1-r2)
    lb_c1_ten = torch.stack(lb_c1).transpose(0, 1)#.view(1,-1)
    lb_c2_ten = torch.stack(lb_c2).transpose(0, 1)#.view(1,-1)
    done1up = lb_c1_ten[:,0] <= state_c1[:,1]
    done1lb = lb_c1_ten[:, 1] >= state_c1[:, 1]

    done_c1 = ~((~done1up)*(~done1lb))

    done2up = lb_c2_ten[:, 0] <= state_c2[:, 1]
    done2lb = lb_c2_ten[:, 1] >= state_c2[:, 1]

    done_c2 = ~((~done2up) * (~done2lb))

    going_out_reward = 1*torch.ones(reward.shape)

    reward = reward - done_c1 * going_out_reward

    reward = reward + done_c2 * going_out_reward

    return reward, -reward, done_c1, done_c2

def getfreezereward(state_c1,state_c2, lb_c1, lb_c2, prev_state_c1, prev_state_c2):
    r1 = state_c1[:, 0] - prev_state_c1[:, 0]
    r2 = state_c2[:, 0] - prev_state_c2[:, 0]
    #reward = state_c1[:,0] - state_c2[:,0]
    reward = (r1-r2)
    lb_c1_ten = torch.stack(lb_c1).transpose(0, 1)#.view(1,-1)
    lb_c2_ten = torch.stack(lb_c2).transpose(0, 1)#.view(1,-1)
    done1up = lb_c1_ten[:,0] <= state_c1[:,1]
    done1lb = lb_c1_ten[:, 1] >= state_c1[:, 1]

    done_c1 = ~((~done1up)*(~done1lb))

    done2up = lb_c2_ten[:, 0] <= state_c2[:, 1]
    done2lb = lb_c2_ten[:, 1] >= state_c2[:, 1]

    done_c2 = ~((~done2up) * (~done2lb))

    return reward, -reward, done_c1, done_c2

def getcollosionreward(state_c1,state_c2, lb_c1, lb_c2, prev_state_c1, prev_state_c2):
    r1 = state_c1[:, 0] - prev_state_c1[:, 0]
    r2 = state_c2[:, 0] - prev_state_c2[:, 0]
    #reward = state_c1[:,0] - state_c2[:,0]
    reward = (r1-r2)
    lb_c1_ten = torch.stack(lb_c1).transpose(0, 1)#.view(1,-1)
    lb_c2_ten = torch.stack(lb_c2).transpose(0, 1)#.view(1,-1)

    long_dist = state_c1[:,0] - state_c2[:,0]
    lat_dist = state_c1[:, 1]- state_c2[:, 1]

    collision_long = torch.abs(long_dist) < (l_f +l_r)
    collision_lat = torch.abs(lat_dist) < W
    collision = collision_lat*collision_long

    c1_coll = collision * (long_dist<0)
    c2_coll = collision * (long_dist >= 0)

    done1up = lb_c1_ten[:,0] <= state_c1[:,1]
    done1lb = lb_c1_ten[:, 1] >= state_c1[:, 1]

    c1_bounds = ~((~done1up)*(~done1lb))
    done_c1 = c1_bounds

    done2up = lb_c2_ten[:, 0] <= state_c2[:, 1]
    done2lb = lb_c2_ten[:, 1] >= state_c2[:, 1]

    c2_bounds = ~((~done2up) * (~done2lb))
    done_c2 = c2_bounds

    collision_reward = -0.5 * torch.ones(reward.shape)
    reward = reward + c1_coll * collision_reward
    reward = reward - c2_coll * collision_reward

    return reward, -reward, done_c1, done_c2

def getNFcollosionreward(state_c1,state_c2, lb_c1, lb_c2, prev_state_c1, prev_state_c2):
    r1 = state_c1[:, 0] - prev_state_c1[:, 0]
    r2 = state_c2[:, 0] - prev_state_c2[:, 0]
    #reward = state_c1[:,0] - state_c2[:,0]
    reward = (r1-r2)
    lb_c1_ten = torch.stack(lb_c1).transpose(0, 1)#.view(1,-1)
    lb_c2_ten = torch.stack(lb_c2).transpose(0, 1)#.view(1,-1)

    long_dist = state_c1[:,0] - state_c2[:,0]
    lat_dist = state_c1[:, 1]- state_c2[:, 1]

    collision_long = torch.abs(long_dist) < (l_f +l_r)
    collision_lat = torch.abs(lat_dist) < W
    collision = collision_lat*collision_long

    c1_coll = collision * (long_dist<0)
    c2_coll = collision * (long_dist >= 0)

    done1up = lb_c1_ten[:,0] <= state_c1[:,1]
    done1lb = lb_c1_ten[:, 1] >= state_c1[:, 1]

    c1_bounds = ~((~done1up)*(~done1lb))
    n_c1 =  torch.sum(torch.ones(state_c1.size(0))*c1_bounds)
    p1 = state_c1[c1_bounds,0]
    state_c1[c1_bounds,:] = torch.zeros(state_c1[c1_bounds,:].shape)
    state_c1[c1_bounds, 0] =p1
    # done_c1 = c1_bounds

    done2up = lb_c2_ten[:, 0] <= state_c2[:, 1]
    done2lb = lb_c2_ten[:, 1] >= state_c2[:, 1]

    c2_bounds = ~((~done2up) * (~done2lb))
    n_c2 =  torch.sum(torch.ones(state_c2.size(0))*c2_bounds)
    p2 = state_c2[c2_bounds, 0]
    state_c2[c2_bounds, :] = torch.zeros(state_c2[c2_bounds, :].shape)
    state_c2[c2_bounds, 0] = p2
    # done_c2 = c2_bounds

    bound_reward = -1 * torch.ones(reward.shape)
    reward = reward + c1_bounds * bound_reward
    reward = reward - c2_bounds * bound_reward

    collision_reward = -0.5 * torch.ones(reward.shape)
    reward = reward + c1_coll * collision_reward
    reward = reward - c2_coll * collision_reward

    reached_c1 = state_c1[:, 0] > 16.5
    reached_c2 = state_c2[:, 0] > 16.5
    reached = ~((~reached_c1)*(~reached_c2))

    done_c1 = reached
    done_c2 = reached


    # done_c1 = ~((~reached)*(~c1_bounds))
    # done_c2 = ~((~reached)*(~c2_bounds))
    temp=done_c1
    done_c1 = ~((~done_c2)*(~done_c1))
    done_c2 = ~((~done_c2) * (~temp))

    reached_reward = 4 * torch.ones(reward.shape)
    reward = reward + reached_c1 * reached_reward
    reward = reward - reached_c2 * reached_reward

    return reward, -reward, done_c1, done_c2, state_c1, state_c2, n_c1, n_c2


def getfreezecollosionreward(state_c1,state_c2, lb_c1, lb_c2, prev_state_c1, prev_state_c2, prev_coll_c1, prev_coll_c2):
    r1 = state_c1[:, 0] - prev_state_c1[:, 0]
    r2 = state_c2[:, 0] - prev_state_c2[:, 0]
    #reward = state_c1[:,0] - state_c2[:,0]
    reward = (r1-r2)
    lb_c1_ten = torch.stack(lb_c1).transpose(0, 1)#.view(1,-1)
    lb_c2_ten = torch.stack(lb_c2).transpose(0, 1)#.view(1,-1)

    long_dist = state_c1[:,0] - state_c2[:,0]
    lat_dist = state_c1[:, 1]- state_c2[:, 1]

    collision_long = torch.abs(long_dist) < (l_f +l_r)
    collision_lat = torch.abs(lat_dist) < W
    collision = collision_lat*collision_long

    c1_coll = collision * (long_dist<0)
    c2_coll = collision * (long_dist >= 0)

    c1_coll = ~((~c1_coll) * (~prev_coll_c1)) #if it collided earlier it should freeze
    c2_coll = ~((~c2_coll) * (~prev_coll_c2)) #if it collided earlier it should freeze

    done1up = lb_c1_ten[:,0] <= state_c1[:,1]
    done1lb = lb_c1_ten[:, 1] >= state_c1[:, 1]

    c1_bounds = ~((~done1up)*(~done1lb))
    done_c1 = ~((~c1_bounds) * (~c1_coll))

    done2up = lb_c2_ten[:, 0] <= state_c2[:, 1]
    done2lb = lb_c2_ten[:, 1] >= state_c2[:, 1]

    c2_bounds = ~((~done2up) * (~done2lb))
    done_c2 = ~((~c2_bounds) * (~c2_coll))

    return reward, -reward, done_c1, done_c2, c1_coll, c2_coll

def getfreezecollosionReachedreward(state_c1,state_c2, lb_c1, lb_c2, prev_state_c1, prev_state_c2, prev_coll_c1, prev_coll_c2):
    r1 = state_c1[:, 0] - prev_state_c1[:, 0]
    r2 = state_c2[:, 0] - prev_state_c2[:, 0]
    #reward = state_c1[:,0] - state_c2[:,0]
    reward = (r1-r2)
    lb_c1_ten = torch.stack(lb_c1).transpose(0, 1)#.view(1,-1)
    lb_c2_ten = torch.stack(lb_c2).transpose(0, 1)#.view(1,-1)

    long_dist = state_c1[:,0] - state_c2[:,0]
    lat_dist = state_c1[:, 1]- state_c2[:, 1]

    collision_long = torch.abs(long_dist) < (l_f +l_r)
    collision_lat = torch.abs(lat_dist) < W
    collision = collision_lat*collision_long

    c1_coll = collision * (long_dist<0)
    c2_coll = collision * (long_dist >= 0)

    c1_coll = ~((~c1_coll) * (~prev_coll_c1)) #if it collided earlier it should freeze
    c2_coll = ~((~c2_coll) * (~prev_coll_c2)) #if it collided earlier it should freeze

    done1up = lb_c1_ten[:,0] <= state_c1[:,1]
    done1lb = lb_c1_ten[:, 1] >= state_c1[:, 1]

    c1_bounds = ~((~done1up)*(~done1lb))
    done_c1 = ~((~c1_bounds) * (~c1_coll))

    done2up = lb_c2_ten[:, 0] <= state_c2[:, 1]
    done2lb = lb_c2_ten[:, 1] >= state_c2[:, 1]

    c2_bounds = ~((~done2up) * (~done2lb))
    done_c2 = ~((~c2_bounds) * (~c2_coll))

    reached_c1 = state_c1[:, 0] > 20
    reached_c2 = state_c2[:, 0] > 20
    reached = ~((~reached_c1)*(~reached_c2))

    done_c1 = ~((~reached) * (~done_c1))
    done_c2 = ~((~reached) * (~done_c2))

    reached_reward = 1 * torch.ones(reward.shape)
    reward = reward + reached_c1 * reached_reward
    reward = reward - reached_c2 * reached_reward

    return reward, -reward, done_c1, done_c2, c1_coll, c2_coll

def getfreezeTimecollosionReachedreward(state_c1,state_c2, lb_c1, lb_c2, prev_state_c1, prev_state_c2, prev_coll_c1, prev_coll_c2, counter1, counter2):
    r1 = state_c1[:, 0] - prev_state_c1[:, 0]
    r2 = state_c2[:, 0] - prev_state_c2[:, 0]
    #reward = state_c1[:,0] - state_c2[:,0]
    reward = (r1-r2)
    lb_c1_ten = torch.stack(lb_c1).transpose(0, 1)#.view(1,-1)
    lb_c2_ten = torch.stack(lb_c2).transpose(0, 1)#.view(1,-1)

    long_dist = state_c1[:,0] - state_c2[:,0]
    lat_dist = state_c1[:, 1]- state_c2[:, 1]

    collision_long = torch.abs(long_dist) < (l_f +l_r)
    collision_lat = torch.abs(lat_dist) < W
    collision = collision_lat*collision_long

    c1_coll = collision * (long_dist<0)
    c2_coll = collision * (long_dist >= 0)
    counter1 = counter1 + prev_coll_c1*torch.ones(reward.shape) #increasing counter if collided
    prev_coll_c1 = (counter1<34) * prev_coll_c1 #if counter is above 34 then reseting
    counter1 = counter1*(counter1<34) # this will reset the counter

    counter2 = counter2 + prev_coll_c2*torch.ones(reward.shape) #increasing counter if collided
    prev_coll_c2 = (counter2<34) * prev_coll_c2 #if counter is above 34 then reseting
    counter2 = counter2*(counter2<34) # this will reset the counter

    # print(counter1, counter2)
    c1_coll = ~((~c1_coll) * (~prev_coll_c1)) #if it collided earlier it should freeze
    c2_coll = ~((~c2_coll) * (~prev_coll_c2)) #if it collided earlier it should freeze

    done1up = lb_c1_ten[:,0] <= state_c1[:,1]
    done1lb = lb_c1_ten[:, 1] >= state_c1[:, 1]

    c1_bounds = ~((~done1up)*(~done1lb))
    done_c1 = ~((~c1_bounds) * (~c1_coll))

    done2up = lb_c2_ten[:, 0] <= state_c2[:, 1]
    done2lb = lb_c2_ten[:, 1] >= state_c2[:, 1]

    c2_bounds = ~((~done2up) * (~done2lb))
    done_c2 = ~((~c2_bounds) * (~c2_coll))

    reached_c1 = state_c1[:, 0] > 20
    reached_c2 = state_c2[:, 0] > 20
    reached = ~((~reached_c1)*(~reached_c2))

    done_c1 = ~((~reached) * (~done_c1))
    done_c2 = ~((~reached) * (~done_c2))

    reached_reward = 1 * torch.ones(reward.shape)
    reward = reward + reached_c1 * reached_reward
    reward = reward - reached_c2 * reached_reward

    return reward, -reward, done_c1, done_c2, c1_coll, c2_coll, counter1, counter2


def getfreezeprogressreward(state_c1,state_c2, lb_c1, lb_c2, prev_state_c1, prev_state_c2):
    # r1 = state_c1[:, 0] - prev_state_c1[:, 0]
    # r2 = state_c2[:, 0] - prev_state_c2[:, 0]
    reward = state_c1[:,0] - state_c2[:,0]
    # reward = (r1-r2)
    lb_c1_ten = torch.stack(lb_c1).transpose(0, 1)#.view(1,-1)
    lb_c2_ten = torch.stack(lb_c2).transpose(0, 1)#.view(1,-1)
    done1up = lb_c1_ten[:,0] <= state_c1[:,1]
    done1lb = lb_c1_ten[:, 1] >= state_c1[:, 1]

    done_c1 = ~((~done1up)*(~done1lb))

    done2up = lb_c2_ten[:, 0] <= state_c2[:, 1]
    done2lb = lb_c2_ten[:, 1] >= state_c2[:, 1]

    done_c2 = ~((~done2up) * (~done2lb))

    return reward, -reward, done_c1, done_c2

def getfreezeProgressReachedreward(state_c1,state_c2, lb_c1, lb_c2, prev_state_c1, prev_state_c2):
    r1 = state_c1[:, 0] - prev_state_c1[:, 0]
    r2 = state_c2[:, 0] - prev_state_c2[:, 0]
    #reward = state_c1[:,0] - state_c2[:,0]
    reward = (r1-r2)
    reached_c1 = state_c1[:,0]>16
    reached_c2 = state_c2[:, 0] > 16
    reached = ~((~reached_c1)*(~reached_c2))

    # reward = (r1-r2)
    s = 0.0
    lb_c1_ten = torch.stack(lb_c1).transpose(0, 1)#.view(1,-1)
    lb_c2_ten = torch.stack(lb_c2).transpose(0, 1)#.view(1,-1)
    done1up = lb_c1_ten[:,0] <= state_c1[:,1] - s
    done1lb = lb_c1_ten[:, 1] >= state_c1[:, 1] + s

    bound_c1 = ~((~done1up)*(~done1lb))
    done_c1 = ~((~reached)*(~bound_c1))

    done2up = lb_c2_ten[:, 0] <= state_c2[:, 1] -s
    done2lb = lb_c2_ten[:, 1] >= state_c2[:, 1] +s

    bound_c2 = ~((~done2up) * (~done2lb))
    done_c2 = ~((~reached)*(~bound_c2))

    reached_reward = 5 * torch.ones(reward.shape)
    reward = reward + reached_c1 * reached_reward
    reward = reward - reached_c2 * reached_reward
    return reward, -reward, done_c1, done_c2

def getdone(state_c1,state_c2):
    return False

def get_reward_single(state_c1,  b_c1, kappa, prev_state_c1):
    # reward = state_c1[:,0]
    # vx = state_c1[:,3]
    # vy = state_c1[:, 4]
    # mu = state_c1[:,2]
    # d = state_c1[:,1]
    # reward = 0.03*(vx * torch.cos(mu) - vy * torch.sin(mu))/ (1.0 - d * kappa)

    reward = state_c1[:,0] - prev_state_c1[:,0]

    lb_c1_ten = torch.stack(b_c1).transpose(0, 1)  # .view(1,-1)
    done1up = lb_c1_ten[:, 0] <= state_c1[:, 1]
    done1lb = lb_c1_ten[:, 1] >= state_c1[:, 1]

    done_c1 = ~((~done1up) * (~done1lb))

    going_out_reward = 1 * torch.ones(reward.shape)

    reward = reward - done_c1 * going_out_reward

    time_reward = 0.01 * torch.ones(reward.shape)

    reward = reward - time_reward

    reached_reward = 1 * torch.ones(reward.shape)

    reached = state_c1[:,0]>=16.4

    reward = reward + reached*reached_reward

    done = ~((~done_c1) * (~reached))

    return reward, done

def get_minibatch(state_mat, action_mat, log_probs, returns, advantage_mat, size_mini_batch):
    size_batch = state_mat.size(0)
    for _ in range(size_batch // size_mini_batch):
        rand_ids = np.random.randint(0, size_batch, size_mini_batch)
        yield state_mat[rand_ids], action_mat[rand_ids], log_probs[rand_ids], returns[rand_ids], advantage_mat[rand_ids].view(-1,1)


def ppo_update(state_mat, action_mat, log_probs, returns, advantage_mat, clip_param, num_ppo_epochs, size_mini_batch,actor_critic, optim_actor_critic, coeff_ent):
    for k in range(num_ppo_epochs):
        for state_mb, action_mb, old_log_probs, return_mb, advantage_mb in get_minibatch(state_mat, action_mat,
                                                                                         log_probs, returns,
                                                                                         advantage_mat,size_mini_batch):
            dist, val = actor_critic(state_mb)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action_mb)
            ratio = (new_log_probs - old_log_probs).exp()
            # print(ratio)

            # print(ratio,advantage_mb)

            surr1 = ratio * advantage_mb
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage_mb

            actor_loss = - torch.min(surr1,
                                     surr2).mean()  # it just means out entire batch, and minimum  is element wise
            critic_loss = (return_mb - val).pow(2).mean()


            for param in actor_critic.critic.parameters():
                critic_loss += param.pow(2).sum() * 1e-3


            loss = 0.5 * critic_loss + actor_loss + coeff_ent * entropy
            #print(loss.detach().numpy(),critic_loss.detach().numpy(),actor_loss,entropy)
            optim_actor_critic.zero_grad()
            loss.backward()

            #torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 40)

            total_norm = 0
            for p in actor_critic.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)

            optim_actor_critic.step()

            yield loss.detach().numpy(), critic_loss.detach().numpy(), actor_loss.detach().numpy(), entropy.detach().numpy(), total_norm


# state_c1 = torch.tensor([1 , -0.3]).view(1,2)
# state_c2 = torch.tensor([0.9,-0.3]).view(1,2)
# lb_c1 =  [torch.tensor([0.4,-0.4])]
# lb_c2 =  [torch.tensor([0.4,-0.4])]
#
# print(getreward(state_c1,state_c2, lb_c1, lb_c2))
