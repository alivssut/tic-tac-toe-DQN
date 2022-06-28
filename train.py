from agent import DQNAgent
from environment import Environment
import numpy as np

EPISODES = 1000

env = Environment()
state_size = env.observation_space
action_size = env.action_space
agent = DQNAgent(state_size + 1, action_size)
done = False
batch_size = 16
agent.load("tictoctoe_weights(5).h5")

print(agent.model.summary())

for e in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    state = np.append(state, [[env.turn]], axis=1)
    for time in range(100):
        action = agent.act(state, env.available())
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        next_state = np.append(next_state, [[env.turn]], axis=1)
        agent.memorize(state, action, reward, next_state, done)
        state = next_state
        env.turn *= -1
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(e, EPISODES, time, agent.epsilon))
            if e % 100 == 0:
                agent.save("tictoctoe_weights.h5")
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
