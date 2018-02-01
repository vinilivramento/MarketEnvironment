from markettrading.environments.env import Env
from markettrading.backtradercomm.backtrader_comm import BacktraderCommunication

class MarketEnvironment(Env):
    action_space = BacktraderCommunication.action_space 
    observation_space = BacktraderCommunication.observation_space
    def __init__(self):
        pass
        # self.reward_range = (-np.inf, np.inf)

    def step(self, action):
        return self._backtrader.step(action)

    def reset(self):
        self._backtrader = BacktraderCommunication()
        return self._backtrader.reset()

    def render(self, mode='human'):
        pass

    def close(self):
        pass
