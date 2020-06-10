import numpy as np

class Blob:
  def __init__(self, size):
    self.size = size
    self.x = np.random.randint(0, size)
    self.y = np.random.randint(0, size)

  def __str__(self):
    return f"Blob ({self.x}, {self.y})"

  def __sub__(self, other):
    return (self.x - other.x, self.y - other.y)

  def __eq__(self, other):
    return self.x == other.x and self.y == other.y

  def action(self, choice):
    if choice == 0:
      self.move(x=1, y=1)
    elif choice == 1:
      self.move(x=-1, y=-1)
    elif choice == 2:
      self.move(x=-1, y=1)
    elif choice == 3:
      self.move(x=1, y=-1)
    elif choice == 4:
      self.move(x=1, y=0)
    elif choice == 5:
      self.move(x=-1, y=0)
    elif choice == 6:
      self.move(x=0, y=1)
    elif choice == 7:
      self.move(x=0, y=-1)
    elif choice == 8:
      self.move(x=0, y=0)

  def move(self, x=False, y=False):
    if not x:
      self.x += np.random.randint(-1, 2)
    else:
      self.x += x
    if not y:
      self.y += np.random.randint(-1, 2)
    else:
      self.y += y

    if self.x < 0:
      self.x = 0
    elif self.x > self.size - 1:
      self.x = self.size - 1

    if self.y < 0:
      self.y = 0
    elif self.y > self.size - 1:
      self.y = self.size - 1