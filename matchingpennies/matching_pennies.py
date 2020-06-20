import numpy as np
import random
class pennies_game():
    def __init__(self):
        self.number_of_players = 2
        self.reward = np.zeros(2)
        self.done = False

    def step(self, action):
        if action[0] == 0:
            if action[1] ==0:
                reward = 1
            else:
                reward = -1
        else:
            if action[1] == 0:
                reward = -1
            else:
                reward = 1

        self.reward = np.array((reward,-reward))
        info = {}
        return 0, self.reward[0], self.reward[1], True, info

    def reset(self):
        #self.state = np.zeros(2)
        self.reward = np.zeros(2)
        self.done = False
        info = {}
        return 0, self.reward[0], self.reward[1], self.done, info


if __name__ == '__main__':
    env = pennies_game()

    for i in range(10):
        state, _, _, _, _ = env.reset()
        action = (np.random.randint(3,size=2)).reshape(2)
        state, reward1, reward2, done, _ =  env.step(action)
        print(action, reward1, reward2)