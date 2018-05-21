import random

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

# actions
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_MOVE_SCREEN = actions.FUNCTIONS.Attack_screen.id

# features AI can see
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index

# values
_BACKGROUND = 0
_PLAYER_SELF = 1
_PLAYER_NEUTRAL = 3
_SELECT_ALL = [0]
_NOT_QUEUED = [0]

# actions
ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_MOVE_SCREEN = 'movescreen'
ACTION_SELECT_POINT = 'selectpoint' # deselect marine

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_ARMY,
    ACTION_MOVE_SCREEN,
    ACTION_SELECT_POINT
]

# Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]

            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)

        q_predict = self.q_table.ix[s, a]
        q_target = r + self.gamma * self.q_table.ix[s_, :].max()

        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))

class SmartAgent(base_agent.BaseAgent):
    def __init__(self):
        super(SmartAgent, self).__init__()

        # initialize Q-learning table with actions list smart_actions
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))

        # track previous action and state
        self.previous_action = None
        self.previous_state = None

    def step(self, obs):
        super(SmartAgent, self).step(obs)

        player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        beacon_x, beacon_y = (obs.observation['screen'][_PLAYER_RELATIVE] == _PLAYER_NEUTRAL).nonzero()

        current_state = [player_x.mean(),
                         player_y.mean(),
                         beacon_x.mean(),
                         beacon_y.mean()]

        if self.previous_action is not None:
            self.qlearn.learn(str(self.previous_state), self.previous_action, obs.reward, str(current_state))

        rl_action = self.qlearn.choose_action(str(current_state))
        smart_action = smart_actions[rl_action]

        self.previous_state = current_state
        self.previous_action = rl_action

        if smart_action == ACTION_DO_NOTHING:
            return actions.FunctionCall(_NO_OP, [])
        elif smart_action == ACTION_SELECT_ARMY:
            if _SELECT_ARMY in obs.observation['available_actions']:
                return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
        elif smart_action == ACTION_SELECT_POINT:
            if _SELECT_POINT in obs.observation['available_actions']:
                background_x, background_y = (obs.observation['screen'][_PLAYER_RELATIVE] == _BACKGROUND).nonzero()
                x_index = np.random.randint(0, len(background_x))
                y_index = np.random.randint(0, len(background_y))
                point_x = background_x[x_index]
                point_y = background_y[y_index]

                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, [point_y, point_x]])
        elif smart_action == ACTION_MOVE_SCREEN:
            if _MOVE_SCREEN in obs.observation["available_actions"]:
                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [beacon_y.mean(), beacon_x.mean()]])

        return actions.FunctionCall(_NO_OP, [])




