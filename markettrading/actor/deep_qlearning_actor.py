import logging
import random
import matplotlib.pyplot as plt
import numpy as np

from collections import deque
from markettrading.environments.market_env import MarketEnvironment
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.optimizers import Adam
    
class Deep_QLearning_Actor:
    def __init__(self, layers_and_act_func, env):
        self._learning_rate = 0.01 
        self._epsilon = 1.0 ## exploration rate
        self._min_epsilon = 0.01 
        self._epsilon_decay = 0.995 
        self._gamma = 0.99 #future discount rate 
        self._batch_size = 64
        self._memory = deque(maxlen=10000)
        self._env = env

        self._num_actions = self._env.action_space.n
        self._state_size = self._env.observation_space#env.observation_space.shape[0], #[o,h,l,c,f_ma, m_ma, s_ma,position] 
      
        self._model = self._build_model(layers_and_act_func)

    def _compile_model(self, model):
        model.compile(loss='mse', optimizer=Adam(lr=self._learning_rate, clipnorm=2))
        return model

    def _build_model(self, layers_and_act_func):
        model = Sequential()
        model.add(Dense(layers_and_act_func[0][0], input_dim=self._state_size, activation=layers_and_act_func[0][1]))
        for i in range(1, len(layers_and_act_func)):
            model.add(Dense(layers_and_act_func[i][0], activation=layers_and_act_func[i][1]))
        model.add(Dense(self._num_actions, activation='linear'))

        self._compile_model(model)
        return model

    def _break_minibatch_into_arrays(self):
        minibatch = random.sample(self._memory, self._batch_size)
        cur_states  = np.array([x[0] for x in minibatch])
        actions     = np.array([x[1] for x in minibatch])
        rewards     = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        dones       = np.array([x[4] for x in minibatch])
        return cur_states, actions, rewards, next_states, dones

    ## Y = r + gamma * argmax_a(Q(s',a; w'))
    ## Loss = (Y - Q(s,a; w))^2
    def _replay(self):
        if len(self._memory) < self._batch_size: return
        cur_states, actions, rewards, next_states, dones = self._break_minibatch_into_arrays()

        Y = self._model.predict(cur_states)
        fut_action = self._model.predict(next_states)
        for i in range(self._batch_size):
            Y[i, actions[i]] = rewards[i] if dones[i] else rewards[i] + self._gamma * np.max(fut_action[i]) 
        history = self._model.fit(cur_states, Y, batch_size=self._batch_size, epochs=1, verbose=0) #single gradient update over one batch of samples
        self._epsilon_scaling()

    def _preprocess(self, state):
        return np.reshape(state, (1, self._state_size))

    def _update(self, state, action, reward, next_state, done):
        self._memory.append([state, action, reward, next_state, done])

    def _epsilon_scaling(self):
        self._epsilon = max(self._min_epsilon, self._epsilon*self._epsilon_decay)

    def act(self, state):
        if np.random.rand() <= self._epsilon:
            return np.random.randint(self._num_actions)
        else:
            return np.argmax(self._model.predict(self._preprocess(state)))

    def train(self, num_episodes=500):
        print("-------------------------Training Actor------------------------")
        max_score = [-1,-1]
        total_reward_per_episode = []
        for episode in range(num_episodes):
            cur_state = self._env.reset()
            done = False
            acc_reward = 0
            while not done:
                action = self.act(cur_state)
                next_state, reward, done, _ = self._env.step(action)
                # print("Action ", action, " Reward ", reward)
                self._update(cur_state, action, reward, next_state, done)
                acc_reward += reward
                cur_state = next_state
            total_reward_per_episode.append(acc_reward)
            logging.info("Episode {}/{}: score: {} epsilon: {:.2}" .format(episode, num_episodes, acc_reward, self._epsilon))
            if acc_reward > max_score[0]: 
                max_score[0] = acc_reward
                max_score[1] = episode
            self._replay()
            # if episode % 10 == 0:
                # self._env.render()
        logging.info("Max Score: {} at episode: {}" .format(max_score[0], max_score[1]))
        self.plot(total_reward_per_episode)

    def test(self, num_episodes=1):
        print("-------------------------Testing Actor------------------------")
        max_score = [-1,-1]
        total_reward_per_episode = []
        self._epsilon = -1.0 ## ensure to always get the highest cost during test mode
        for episode in range(num_episodes):
            cur_state = self._env.reset()
            done = False
            acc_reward = 0
            while not done:
                action = self.act(cur_state)
                next_state, reward, done, _ = self._env.step(action)
                acc_reward += reward
                cur_state = next_state
            total_reward_per_episode.append(acc_reward)
            logging.info("Episode {}/{}: score: {} epsilon: {:.2}" .format(episode, num_episodes, acc_reward, self._epsilon))
            if acc_reward > max_score[0]: 
                max_score[0] = acc_reward
                max_score[1] = episode
            self._env.render()
        logging.info("Max Score: {} at episode: {}" .format(max_score[0], max_score[1]))

    def plot(self, total_reward_per_episode):
        fig = plt.figure('training')
        plt.plot(total_reward_per_episode)
        plt.ylabel('episode')
        plt.xlabel('reward')
        plt.show()

