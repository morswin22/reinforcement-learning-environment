import random
import time
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import (Activation, Conv2D, Dense, Dropout,
                                     Flatten, MaxPooling2D)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

class ModifiedTensorBoard(TensorBoard):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.step = 1
    self._log_write_dir = self.log_dir
    self.writer = tf.summary.create_file_writer(self.log_dir)

  def set_model(self, model):
    pass

  def on_epoch_end(self, epoch, logs=None):
    self.update_stats(**logs)

  def on_batch_end(self, batch, logs=None):
    pass

  def on_train_end(self, _):
    pass

  def update_stats(self, **stats):
    self._log_metrics(stats, DQN.MODEL_NAME, self.step)

class DQN:
  REPLAY_MEMORY_SIZE = 50_000
  MIN_REPLAY_MEMORY_SIZE = 1_000
  MODEL_NAME = "256x2"
  MINIBATCH_SIZE = 64
  DISCOUNT = 0.99
  UPDATE_TARGET_EVERY = 5

  def __init__(self, env):
    self.observation_space_values = env.OBSERVATION_SPACE_VALUES
    self.action_space_size = env.ACTION_SPACE_SIZE

    # main model
    self.model = self.create_model()

    # target model
    self.target_model = self.create_model()
    self.target_model.set_weights(self.model.get_weights())

    self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)
    self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{self.MODEL_NAME}-{int(time.time())}")
    self.target_update_counter = 0

  def create_model(self):
    model = Sequential()
    model.add(Conv2D(256, (3, 3), input_shape=self.observation_space_values))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(64))

    model.add(Dense(self.action_space_size, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
    return model

  def load_model(self, path):
    self.model = load_model(path)
    self.target_model = load_model(path)

  def update_replay_memory(self, transition):
    self.replay_memory.append(transition)

  def get_qs(self, state):
    return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

  def train(self, terminal_state, step):
    if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
      return

    minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)

    current_states = np.array([transition[0] for transition in minibatch]) / 255
    current_qs_list = self.model.predict(current_states)

    new_current_states = np.array([transition[3] for transition in minibatch]) / 255
    future_qs_list = self.target_model.predict(new_current_states)

    X = []
    y = []

    for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
      if not done:
        max_future_q = np.max(future_qs_list[index])
        new_q = reward + self.DISCOUNT * max_future_q
      else:
        new_q = reward

      current_qs = current_qs_list[index]
      current_qs[action] = new_q

      X.append(current_state)
      y.append(current_qs)

    self.model.fit(np.array(X) / 255, np.array(y), batch_size=self.MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

    if terminal_state:
      self.target_update_counter += 1

    if self.target_update_counter > self.UPDATE_TARGET_EVERY:
      self.target_model.set_weights(self.model.get_weights())
      self.target_update_counter = 0
