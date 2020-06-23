import json
import os
import time
from collections import deque

import numpy as np
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

episode_rewards = deque(maxlen=AGGREGATE_STATS_EVERY)

start_episode = 0

if os.path.isdir('paused'):
  is_paused = list(os.scandir('paused')).pop()
  if is_paused:
    with open(os.path.join(is_paused, 'config.json'), 'r') as paused_config:
      data = json.load(paused_config)
      print("Would you like to resume training of latest paused model? [Y/n]")
      print(f"Model name: {data['name']}")
      print(f"Paused at {is_paused.name}")
      print(f"Episode #{data['episode']}")
      answer = input()
      if answer == "" or answer.lower() == "y":
        start_episode = data['episode']
        epsilon = data['epsilon']
        for reward in data['episode_rewards']:
          episode_rewards.append(reward)
        agent.load_model(os.path.join(is_paused, data['name']))
        agent.model.summary()

try:
  for episode in tqdm(range(start_episode+1, EPISODES+1), ascii=True, unit="episode"):
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
      average_reward = sum(episode_rewards) / AGGREGATE_STATS_EVERY
      min_reward = min(episode_rewards)
      max_reward = max(episode_rewards)
      agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

      if min_reward >= MIN_REWARD:
        agent.model.save(f"models/{agent.MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model")

    if epsilon > MIN_EPSILON:
      epsilon *= EPSILON_DECAY
      epsilon = max(MIN_EPSILON, epsilon)

except KeyboardInterrupt:
  base = f"paused/{time.strftime('%Y-%m-%d %Hh%Mm%Ss')}"
  os.makedirs(base, exist_ok=True)
  agent.model.save(f"{base}/model_{agent.MODEL_NAME}")
  with open(f"{base}/config.json", 'w') as file:
    json.dump({'name': f"model_{agent.MODEL_NAME}", 'epsilon': epsilon, 'episode': episode, 'episode_rewards': list(episode_rewards)}, file)

else:
  agent.model.save(f"models/{agent.MODEL_NAME}__finished__{int(time.time())}.model")