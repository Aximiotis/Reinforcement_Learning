import numpy as np

class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def restart(self, x, y):
        self.x = 0
        self.y = 0

    def get_position(self):
        return self.x, self.y

    def set_position(self, x, y):
        self.x = x
        self.y = y

    def move(self, next_x, next_y):
        self.x = next_x
        self.y = next_y

    def Identify_Obstacle(self, grid):
        if grid[self.y][self.x] == -10:
            return True
        return False