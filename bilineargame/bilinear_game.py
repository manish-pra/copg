import numpy as np
import random

class bilinear():
    def __init__(self):
        self.number_of_players = 2
        self.reward = np.zeros(2)
        self.done = False

    def step(self, action):
        self.reward = np.array((action[0]*action[1],action[0]*action[1]))
        info = {}
        return 0, self.reward[0], self.reward[1], True, info

    def reset(self):
        #self.state = np.zeros(2)
        self.reward = np.zeros(2)
        self.done = False
        info = {}
        return 0, self.reward[0], self.reward[1], self.done, info


if __name__ == '__main__':
    env = bilinear()

    for i in range(10):
        state, _, _, _, _ = env.reset()
        state, reward1, reward2, done, _ =  env.step((np.random.randint(2,size=2)).reshape(2))
        print(state, reward1, reward2 , done)