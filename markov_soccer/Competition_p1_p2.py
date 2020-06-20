# Game imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0,'..')
from open_spiel.python import rl_environment

import torch
from torch.distributions import Categorical

import numpy as np

from markov_soccer.networks import policy
from markov_soccer.networks import critic
import matplotlib.pyplot as plt

from copg_optim.critic_functions import critic_update, get_advantage
from markov_soccer.soccer_state import get_relative_state, get_two_state

import scipy.io

p1 = policy(12,5)
p2 = policy(12,5)

num_episode = 2000

game = "markov_soccer"
num_players = 2

env = rl_environment.Environment(game)
num_actions = env.action_spec()["num_actions"]
print(num_actions)

player1 = 'LOLA_P1'#'CoPG_P1'
player2 = 'LOLA_P2'#'TRCoPO_P2'

p1.load_state_dict(torch.load("pretrained_models/" + player1 + ".pth"))
p2.load_state_dict(torch.load("pretrained_models/" + player2 + ".pth"))
win_mat=[]

ball_status = [] #0 ball with A and A scored, 1, ball with B and b scored, 2 is ball with A and B scored, 3 is ball with B and A scored
originality_status = []
game_length = []
for t_eps in range(num_episode):
    mat_action = []

    mat_state1 = []
    mat_reward1 = []
    mat_done = []

    mat_state2 = []
    mat_reward2 = []
    time_step = env.reset()
    _,_,rel_state = get_relative_state(time_step)
    pre_a_status, pre_b_status, rel_state1, rel_state2 = get_two_state(time_step)
    state = time_step.observations["info_state"][0]
    i=0
    if t_eps % 100 == 0:
        win_array = np.array(win_mat)
        print(np.sum(win_array == 0), np.sum(win_array == 1), np.sum(win_array == 2))
    if t_eps == num_episode -1:
        fig, ax = plt.subplots()
        data = np.bincount(ball_status)/np.sum(np.bincount(ball_status))
        if data[0] == 0:
            ax.bar(['$A_1$','$B_1$','$A_2$','$B_2$','$A_3$','$B_3$','$A_4$','$B_4$'], data[1:9], width=0.9, align='center')
        else:
            ax.bar(['N','$A_1$', '$B_1$', '$A_2$', '$B_2$', '$A_3$', '$B_3$', '$A_4$', '$B_4$'], data[0:9], width=0.9,
                   align='center')
        plt.show()
    temp_ball_stat = 0
    originality = 0

    #data_collection
    while time_step.last()==False:
        action1_prob = p1(torch.FloatTensor(rel_state1))
        # print('a1',action1_prob)
        # action1 = np.argmax(action1_prob.detach().numpy())
        dist1 = Categorical(action1_prob)
        action1 = dist1.sample() # it will learn probablity for actions and output 0,1,2,3 as possibility of actions
        # print(action1)

        action2_prob = p2(torch.FloatTensor(rel_state2))
        # action2 = np.argmax(action2_prob.detach().numpy())
        # print('a2',action2_prob)
        dist2 = Categorical(action2_prob)
        action2 = dist2.sample()

        action = np.array([action1, action2])

        mat_state1.append(torch.FloatTensor(rel_state1))
        mat_state2.append(torch.FloatTensor(rel_state2))
        mat_action.append(torch.FloatTensor(action))

        time_step = env.step(action)
        _,_,rel_state = get_relative_state(time_step)
        a_status, b_status, rel_state1, rel_state2 = get_two_state(time_step)

        if pre_a_status ==0 and a_status ==1:
            #print('a got the ball')
            if temp_ball_stat%2==0 and temp_ball_stat!=0: # B had the ball, but A took it
                temp_ball_stat = temp_ball_stat + 1
            else:
                temp_ball_stat = 1 # A got the ball
                originality = 1

        if pre_b_status ==0 and b_status ==1:
            #print('b got the ball')
            if temp_ball_stat%2==1:
                temp_ball_stat = temp_ball_stat + 1 # B snatched the ball
            else:
                temp_ball_stat = 2
                originality = 2

        pre_a_status = a_status
        pre_b_status = b_status

        # print(pretty_board(time_step))
        reward1 = time_step.rewards[0]
        reward2 = time_step.rewards[1]
        mat_reward1.append(torch.FloatTensor([reward1]))
        mat_reward2.append(torch.FloatTensor([reward2]))

        if time_step.last() == True:
            done = True
        else:
            done=False

        mat_done.append(torch.FloatTensor([1 - done]))
        # if t_eps > 4990:
        #     print(pretty_board(time_step))
        i=i+1
        if done == True:
            temp=0
            if reward1>0:
                temp=1
                #print("a won")
            if reward2>0:
                temp=2
                #print("b won")
            if originality ==1: #A got the ball initially
                if temp_ball_stat>1: # Finally B scored the goal or A scored the goal after taking ball from b
                    temp_ball_stat = temp_ball_stat + 2
            if temp ==0 and temp_ball_stat is not 0:
                temp_ball_stat = 0 # no one score goal
            ball_status.append(temp_ball_stat)
            originality_status.append(originality)
            game_length.append(i)
            #print(t_eps, i, "done reached")

            win_mat.append(temp)
            #print(state1, state2, reward1,reward2, done)
            break

# To save data
# scipy.io.savemat('../'+ 'p2_soccer_' + player1 + 'vs' + player2 +'.mat', mdict={'ball_status': ball_status,'win_mat':win_mat, 'originality':originality_status, 'game_length':game_length})
# print(np.sum(win_array==0))
# print(np.sum(win_array==1))
# print(np.sum(win_array==2))