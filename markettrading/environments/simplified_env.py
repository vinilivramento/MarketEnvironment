import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gym import spaces
from markettrading.environments.actions import Action, Position
from markettrading.environments.env import Env
from matplotlib.pyplot import cm
from sklearn import preprocessing

class SimplifiedMarketEnv(Env):
    action_space = spaces.Discrete(Action.size())
    observation_space = 2

    def __init__(self, **kwargs):
        """
        Keyword Args:
            
            price_function = Linear               Linear, Sinoid
        """

        self._time_steps = np.arange(0, 101, 1)
        if 'price_function' in kwargs and kwargs['price_function'] == 'Sinoid':
            self._time_steps = np.sin(self._time_steps)+1

        self._time_steps = pd.Series(self._time_steps)
        mov_avg = np.nan_to_num(self._time_steps.rolling(2).mean())
        self._time_steps = np.column_stack((self._time_steps, mov_avg))

        self._scaler = preprocessing.StandardScaler()
        self._time_steps = self._scaler.fit_transform(self._time_steps)
        
        self._last_time_step = self._time_steps.shape[0] - 1
        self._stake = 1

    def _insert_executed_action(self):
        if self._position == Position.IDLE:
            self._executed_actions.append(self._action.value)
        elif self._position == Position.LONG:
            if self._action == Action.SELL: self._executed_actions.append(self._action.value)
            else                          : self._executed_actions.append(Action.HOLD.value)
        elif self._position == Position.SHORT:
            if self._action == Action.BUY: self._executed_actions.append(self._action.value)
            else                          : self._executed_actions.append(Action.HOLD.value)

    def _compute_reward(self):
        reward = 0
        if self._position == Position.IDLE:
            if self._action == Action.BUY:    self._position = Position.LONG
            elif self._action == Action.SELL: self._position = Position.SHORT
            if self._action != Action.HOLD:  self._position_entry_price = self._price
        elif self._position == Position.LONG:
            if self._action == Action.SELL: 
                reward = self._stake*(self._price - self._position_entry_price)
                self._position = Position.IDLE
                self._position_entry_price = None
            else:
                reward = self._stake*(self._price - self._position_entry_price)
        elif self._position == Position.SHORT:
            if self._action == Action.BUY:
                reward = self._stake*(self._position_entry_price - self._price)
                self._position = Position.IDLE
                self._position_entry_price = None
            else:
                reward = self._stake*(self._position_entry_price - self._price)
        return reward 

    def _update_done(self):
        self._done = self._cur_time_step == self._last_time_step
        if self._done and self._position != Position.IDLE: #close any remaining position
            self._action = Action.SELL if self._position == Position.LONG else Action.BUY
        
    def _next_state(self):
        next_state = self._time_steps[self._cur_time_step : self._cur_time_step+1][0] 
        self._price = next_state[0]
        return next_state

    def step(self, action):
        self._cur_time_step += 1
        self._action = Action(action)
        self._update_done()
        self._insert_executed_action()
        return self._next_state(), self._compute_reward(), self._done, None

    def reset(self):
        self._executed_actions = [0]
        self._cur_time_step = 0
        self._position = Position.IDLE
        self._action = Action.HOLD 
        self._position_entry_price = None
        return self._next_state()

    def render(self, mode='human'):
        actions_array = np.array(self._executed_actions)
        colors = np.where(actions_array==1,'blue', np.where(actions_array==2,'red','black'))
        markers = np.where(actions_array==1,'^', np.where(actions_array==2,'v',''))

        fig = plt.figure('actions')
        y = self._scaler.inverse_transform(self._time_steps)[:,0] #prices
        x = np.arange(len(y))
        plt.plot(x, y)
        for x_, y_, color, marker in zip(x, y, colors, markers):
            plt.scatter(x_, y_, color=color, marker=marker, s=50)
        
        plt.ylabel('price')
        plt.xlabel('time step')
        plt.ylim(min(y)-10,max(y)+10)
        # plt.xticks(x)
        plt.xlim(min(x)-1,max(x)+1)
        plt.show()

    def close(self):
        pass
