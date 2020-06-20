# LQ game having a bilinear rewards

import numpy as np
import torch
import random

class LQ_game():
    def __init__(self):
        self.number_of_players = 2
        self.d = 1
        self.n = 1
        self.state = torch.ones((self.n,1))
        self.reward = np.zeros((2,1))
        self.A = torch.FloatTensor([0.9]).reshape((self.n, self.n))
        self.B1 = torch.FloatTensor([0.8]).reshape((self.n, self.d))
        self.B2 = torch.FloatTensor([1.5]).reshape((self.n, self.d))
        self.W12 = torch.FloatTensor([1]).reshape((self.d, self.d))
        self.W21 = torch.FloatTensor([1]).reshape((self.d, self.d))
        # self.A = np.array([[1,1],[1,2]]).reshape((self.n,self.n))
        # self.B1 = np.array([[1,1], [1,1]]).reshape((self.n,self.d))
        # self.B2 = np.array([[1,1], [1,1]]).reshape((self.n,self.d))
        # self.W12 = np.array([[1, 0], [0, 1]]).reshape((self.d,self.d))
        # self.W21 = np.array([[1, 0], [0, 1]]).reshape((self.d,self.d))
        self.done = False

    def step(self, action1, action2):
        action1 = action1.reshape((self.d,1))
        action2 = action2.reshape((self.d,1))

        self.state = self.A*self.state + self.B1*action1 - self.B2*action2
        r0 = -1*action1*action1 + 1*action2*action2 + self.state*self.state #action1.transpose(0,1)*self.W12*action2.reshape(1)
        self.reward = np.array((r0,r0))
        info = {}
        return self.state, self.reward[0], self.reward[1], False, info

    def reset(self):
        self.state = torch.zeros((self.n,1))
        self.reward = np.zeros((2,1))
        self.done = False
        info = {}
        return self.state, self.reward[0], self.reward[1], self.done, info


if __name__ == '__main__':
    env = LQ_bilinear()
    state, _, _, _, _ = env.reset()

    for i in range(10):
        action1 = np.array(1).reshape(1)
        action2 = np.array(1).reshape(1)

        state, reward1, reward2, done, _ =  env.step(action1,action2)
        print(state, reward1, reward2 , done)