import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='sigmoid'))
        model.compile(loss='mse',
                      optimizer='adam')
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, available_state):
        if np.random.rand() <= self.epsilon:
            if np.squeeze(np.argwhere(available_state == True)).size > 1:
                return random.choice(np.squeeze(np.argwhere(available_state == True)))
            else:
                return np.squeeze(np.argwhere(available_state == True))

        act_values = self.model.predict(state)
        act_values[0][available_state == False] = -2
        return np.argmax(act_values[0])  # returns action

    def act_network(self, state, available_state):
        act_values = self.model.predict(state)
        act_values[0][available_state == False] = -2
        print(act_values[0])
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
