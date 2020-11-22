import gym
import random
import numpy as np

from gym import spaces

def coin_flip(prob_of_head = 0.4):
    '''
    Args:
        prob_of_head - probability of getting a Head on coin flip
    Returns:
        0 for Heads or 1 for Tails
    '''
    return 0 if random.random() < prob_of_head else 1

'''
Represents a Gambler's problem Gym Environment which provides a Fully observable MDP
'''
class GamblersEnv(gym.Env):
    '''
    GamblerEnv represents the Gym Environment for the Gambler's problem environment
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self, initial_cash_in_hand = 50, p_h = 0.40, goal_cash = 100):
        '''
        Constructor for the GamblersEnv class
        Args:
            initial_cash_in_hand - represents the cash that the player has initially
            prob_head - probability of getting a heads on a coin flip
            goal_cash - maximum cash obtained before the game ends
        '''
        self.initial_cash_in_hand = initial_cash_in_hand
        self.cash_in_hand = initial_cash_in_hand
        self.prob_head = p_h
        self.goal_cash = goal_cash
        self.nS = self.goal_cash + 1
        self.nA = min(self.cash_in_hand, 100 - self.cash_in_hand) + 1
        self.reset()

    def reset(self):
        '''
        Resets the environment
        Returns:
            observations containing player's current cash in hand
        '''
        self.cash_in_hand = self.initial_cash_in_hand
        self.nA = min(self.cash_in_hand, 100 - self.cash_in_hand) + 1
        return self.get_obs()

    def get_obs(self):
        '''
        Returns the player's cash in hand as the observation of the environment
        '''
        return (self.cash_in_hand)

    def render(self, mode='human'):
        '''
        Renders the environment
        '''
        print("Current capital: {}".format(self.cash_in_hand))

    def sample_action(self):
        return random.randint(0, self.cash_in_hand)

    def P(self, state, action):
        state += 1
        action += 1
        next_state_win = state + action
        next_state_lose = state - action

        prob_win = self.prob_head
        prob_lose = 1.0 - self.prob_head

        if next_state_win >= self.goal_cash:
            reward_win = 1
            done_win = True

        else:
            reward_win = 0
            done_win = False


        if next_state_lose >= self.goal_cash:
            reward_lose = 1.0
            done_lose = True

        elif next_state_lose <= 0:
            done = True
            reward = 0

        else:
            reward_lose = 0.0
            done_lose = False


        next_SAR = []
        next_state_win = np.clip(next_state_win - 1, 0, self.goal_cash)
        next_state_lose = np.clip(next_state_lose - 1, 0, self.goal_cash)
        next_SAR.append((prob_win, next_state_win, reward_win, done_win))
        next_SAR.append((prob_lose, next_state_lose, reward_lose, done_lose))

        return next_SAR


    def step(self, action):
        '''
        Performs the given action
        Args:
            action : action from the action_space to be taking in the environment
        Returns:
            observation - returns current cash in hand of the player
            reward - reward obtained after taking the given action
            done - True if the episode is complete else False
        '''
        action += 1

        if action > self.cash_in_hand:
            action = self.cash_in_hand

        coinflip_result = coin_flip(self.prob_head)

        if coinflip_result:
            self.cash_in_hand += action
        else:
            self.cash_in_hand -= action

        #self.nA = self.cash_in_hand + 1

        if self.cash_in_hand >= self.goal_cash:
            done = True
            reward = 1

        elif self.cash_in_hand <= 0:
            done = True
            reward = 0

        else:
            done = False
            reward = 0

        self.cash_in_hand = np.clip(self.cash_in_hand, 0, self.goal_cash)

        return (self.cash_in_hand, reward, done, action)
