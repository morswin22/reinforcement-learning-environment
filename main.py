import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import os
from environment import Environment

style.use("ggplot")

LEARNING_RATE = 0.1
DISCOUNT = 0.95

HM_EPISODES = 25_000
SHOW_EVERY = 3_000
SHOW_PREVIEW = True

epsilon = 0.9
EPS_DECAY = 0.9998

start_q_table = None # or filename

env = Environment()

if start_q_table is None:
  q_table = {}
  for x1 in range(-env.SIZE + 1, env.SIZE):
    for y1 in range(-env.SIZE + 1, env.SIZE):
      for x2 in range(-env.SIZE + 1, env.SIZE):
        for y2 in range(-env.SIZE + 1, env.SIZE):
          q_table[(x1, y1, x2, y2)] = np.random.uniform(-6, 0, size = env.ACTION_SPACE_SIZE)
else:
  with open(start_q_table, "rb") as f:
    q_table = pickle.load(f)

episode_rewards = []

for episode in range(HM_EPISODES):
  if not episode % SHOW_EVERY:
    print(f"on #{episode}, epsilon: {epsilon}")
    print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
    show = True
  else:
    show = False

  episode_reward = 0

  current_state = env.reset()

  done = False
  while not done:
    if np.random.random() > epsilon:
      action = np.argmax(q_table[current_state])
    else:
      action = np.random.randint(0, 4)
    
    new_state, reward, done = env.step(action)

    episode_reward += reward

    if SHOW_PREVIEW and show:
      env.render()

    max_future_q = np.max(q_table[new_state])
    current_q = q_table[current_state][action]

    if reward == env.FOOD_REWARD:
      new_q = env.FOOD_REWARD
    elif reward == -env.ENEMY_PENALTY:
      new_q = -env.ENEMY_PENALTY
    else:
      new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

    q_table[current_state][action] = new_q

    current_state = new_state

    episode_reward += reward

  episode_rewards.append(episode_reward)
  epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY, )) / SHOW_EVERY, mode="valid")

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

fname = f"qtables/qtable-{int(time.time())}.pickle"
os.makedirs(os.path.dirname(fname), exist_ok=True)
with open(fname, "wb") as f:
  pickle.dump(q_table, f)