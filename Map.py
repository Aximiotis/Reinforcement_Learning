import numpy as np
import random

class Map():
    def __init__(self, width, height ,MaxObstacles):
        self.width = width
        self.height = height
        self.MaxObstacles = MaxObstacles  
        self.grid = [[-1 for _ in range(width)] for _ in range(height)]

    def get_map(self):

        widths = [random.randint(0, self.width - 1) for _ in range(self.MaxObstacles)]
        heights = [random.randint(0, self.height - 1) for _ in range(self.MaxObstacles)]

        for i in range(self.MaxObstacles):
            self.grid[heights[i]][widths[i]] = -10

        return self.grid

    def set_target(self, target_x, target_y):
        self.grid[target_y][target_x] = 100

    def get_ascii_map(self):
        ascii_map = []
        for row in self.grid:
            ascii_row = []
            for cell in row:
                if cell == -1:
                    ascii_row.append('.')
                elif cell == -10:
                    ascii_row.append('#')
                elif cell == 10:
                    ascii_row.append('T')
                else:
                    ascii_row.append('?')  # Unknown cell value
            ascii_map.append(' '.join(ascii_row))
        return '\n'.join(ascii_map)