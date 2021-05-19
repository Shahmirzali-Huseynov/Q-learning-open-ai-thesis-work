import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 1
        self.gamma = 0.9
        self.alpha = 0.2
        self.episode = 0
        print("gamma: "+str(self.gamma), " || epsilon: "+str(self.epsilon), " || alpha: "+str(self.alpha))

    def select_action(self, state):

        probability = np.ones(self.nA) * self.epsilon/self.nA
        # if state in self.Q:
        # probability *= self.epsilon
        probability[np.argmax(self.Q[state])] += 1 - self.epsilon
        return np.random.choice(self.nA, p=probability)

    def greedy_action(self, state):
        return np.argmax(self.Q[state])

    def step(self, state, action, reward, next_state, done):
        target_reward = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state][action] += self.alpha * (target_reward - self.Q[state][action])
        if done:
            self.episode += 1
            self.epsilon = 1/self.episode