import numpy as np
import time
import os
from tqdm import tqdm
from environment import Environment
from network import DQN

MIN_REWARD = -100

EPISODES = 20_000
AGGREGATE_STATS_EVERY = 50
SHOW_EVERY = 1_000

epsilon = 1
MIN_EPSILON = 0.001
EPSILON_DECAY = MIN_EPSILON ** (2/EPISODES)

env = Environment()
env.RETURN_IMAGES = True

agent = DQN(env)

if not os.path.isdir('models'):
  os.makedirs('models')

episode_rewards = [MIN_REWARD]

for episode in tqdm(range(1, EPISODES+1), ascii=True, unit="episode"):
  show = SHOW_EVERY and episode % SHOW_EVERY == 0

  agent.tensorboard.step = episode
  step = 1
  episode_reward = 0

  current_state = env.reset()

  done = False
  while not done:
    if np.random.random() > epsilon:
      action = np.argmax(agent.get_qs(current_state))
    else:
      action = np.random.randint(0, env.ACTION_SPACE_SIZE)
    
    new_state, reward, done = env.step(action)

    episode_reward += reward

    if show:
      env.render()

    agent.update_replay_memory((current_state, action, reward, new_state, done))
    agent.train(done, step)

    current_state = new_state
    step += 1

  episode_rewards.append(episode_reward)
  if not episode % AGGREGATE_STATS_EVERY or episode == 1:
    average_reward = sum(episode_rewards[-AGGREGATE_STATS_EVERY:])/len(episode_rewards)
    min_reward = min(episode_rewards[-AGGREGATE_STATS_EVERY:])
    max_reward = max(episode_rewards[-AGGREGATE_STATS_EVERY:])
    agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

    if min_reward >= MIN_REWARD:
      agent.model.save(f"models/{agent.MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model")

  if epsilon > MIN_EPSILON:
    epsilon *= EPSILON_DECAY
    epsilon = max(MIN_EPSILON, epsilon)