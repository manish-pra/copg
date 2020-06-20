import numpy as np
import random
class rps_game():
    def __init__(self):
        self.number_of_players = 2
        #self.state = np.zeros(2)
        self.reward = np.zeros(2)
        self.done = False

    def step(self, action):
        #self.state = action#np.array((action[0]*action[1],action[0]*action[1]))
        if action[0] == 0:
            if action[1] ==0:
                reward = 0
            elif action[1] ==1:
                reward = 1
            else:
                reward = -1
        elif action[0] == 1:
            if action[1] == 0:
                reward = -1
            elif action[1] ==1:
                reward = 0
            else:
                reward = 1
        else:
            if action[1] == 0:
                reward = 1
            elif action[1] ==1:
                reward = -1
            else:
                reward = 0

        self.reward = np.array((-reward,reward))
        info = {}
        return 0, self.reward[0], self.reward[1], True, info

    def reset(self):
        #self.state = np.zeros(2)
        self.reward = np.zeros(2)
        self.done = False
        info = {}
        return 0, self.reward[0], self.reward[1], self.done, info

class nz_rps_game():
    def __init__(self):
        self.number_of_players = 2
        #self.state = np.zeros(2)
        self.reward = np.zeros(2)
        self.done = False

    def step(self, action):
        #self.state = action#np.array((action[0]*action[1],action[0]*action[1]))
        if action[0] == 0:
            if action[1] ==0:
                reward1 = 0
                reward2 = 0
            elif action[1] ==1:
                reward1 = 2
                reward2 = -1
            else:
                reward1 = 1
                reward2 = -2
        elif action[0] == 1:
            if action[1] == 0:
                reward1 = 1
                reward2 = -2
            elif action[1] ==1:
                reward1 = 0
                reward2 = 0
            else:
                reward1 = 2
                reward2 = -1

        else:
            if action[1] == 0:
                reward1 = 2
                reward2 = -1
            elif action[1] ==1:
                reward1 = 1
                reward2 = -2
            else:
                reward1 = 0
                reward2 = 0

        self.reward = np.array((reward1,reward2))
        info = {}
        return 0, self.reward[0], self.reward[1], True, info

    def reset(self):
        #self.state = np.zeros(2)
        self.reward = np.zeros(2)
        self.done = False
        info = {}
        return 0, self.reward[0], self.reward[1], self.done, info

if __name__ == '__main__':
    env = rps_game()

    for i in range(10):
        state, _, _, _, _ = env.reset()
        action = (np.random.randint(3,size=2)).reshape(2)
        state, reward1, reward2, done, _ =  env.step(action)
        print(action, reward1, reward2)