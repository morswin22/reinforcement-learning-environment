import numpy as np
from PIL import Image
import cv2
from blob import Blob

class Environment:
  SIZE = 10
  RETURN_IMAGES = False
  MOVE_PENALTY = 1
  ENEMY_PENALTY = 300
  FOOD_REWARD = 25
  OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)
  ACTION_SPACE_SIZE = 9
  PLAYER_N = 1
  FOOD_N = 2
  ENEMY_N = 3
  d = {
    1: (255, 175, 0),
    2: (0, 255, 0),
    3: (0, 0, 255)
  }

  def reset(self):
    self.player = Blob(self.SIZE)
    self.food = Blob(self.SIZE)
    while self.food == self.player:
      self.food = Blob(self.SIZE)
    self.enemy = Blob(self.SIZE)
    while self.enemy == self.player or self.enemy == self.food:
      self.enemy = Blob(self.SIZE)

    self.episode_step = 0

    if self.RETURN_IMAGES:
      observation = np.array(self.get_image())
    else:
      observation = ((self.player - self.food) + (self.player - self.enemy))
    
    return observation

  def step(self, action):
    self.episode_step += 1
    self.player.action(action)

    # self.enemy.move()
    # self.food.move()

    if self.RETURN_IMAGES:
      new_observation = np.array(self.get_image())
    else:
      new_observation = ((self.player - self.food) + (self.player - self.enemy))

    if self.player == self.enemy:
      reward = -self.ENEMY_PENALTY
    elif self.player == self.food:
      reward = self.FOOD_REWARD
    else:
      reward = -self.MOVE_PENALTY

    done = False
    if reward == self.FOOD_REWARD or reward == -self.ENEMY_PENALTY or self.episode_step >= 200:
      done = True

    return new_observation, reward, done

  def render(self):
    img = self.get_image()
    img = img.resize((300, 300))
    cv2.imshow("image", np.array(img))
    cv2.waitKey(1)

  def get_image(self):
    env = np.zeros(self.OBSERVATION_SPACE_VALUES, dtype=np.uint8) 
    env[self.food.y][self.food.x] = self.d[self.FOOD_N]
    env[self.enemy.y][self.enemy.x] = self.d[self.ENEMY_N]
    env[self.player.y][self.player.x] = self.d[self.PLAYER_N]
    img = Image.fromarray(env, 'RGB')
    return img