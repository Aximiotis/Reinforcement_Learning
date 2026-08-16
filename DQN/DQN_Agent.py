import numpy as np


class Agent:

    def __init__(self, num_states, num_actions, grid_width, grid_height):

        self.states = num_states
        self.actions = num_actions

        # Learning parameters
        self.learning_rate = 0.1      # alpha
        self.discount_rate = 0.9      # gamma

        # Grid dimensions
        self.grid_width = grid_width
        self.grid_height = grid_height


    def set_learning_rates(self, alpha, gamma):

        self.learning_rate = alpha
        self.discount_rate = gamma



    def get_coordinates(self, state):

        y = state // self.grid_width
        x = state % self.grid_width

        return x, y


    # ---------------------------------------------------------
    # ACTION MASK
    # ---------------------------------------------------------
    #
    # 0 = Up
    # 1 = Down
    # 2 = Left
    # 3 = Right
    #
    # True  -> valid action
    # False -> invalid action
    # ---------------------------------------------------------

    def get_action_mask(self, x, y):

        mask = [True, True, True, True]

        # Up
        if y == 0:
            mask[0] = False

        # Down
        if y == self.grid_height - 1:
            mask[1] = False

        # Left
        if x == 0:
            mask[2] = False

        # Right
        if x == self.grid_width - 1:
            mask[3] = False

        return mask


    # ---------------------------------------------------------
    # CHOOSE ACTION
    # ---------------------------------------------------------

    def choose_action(self, x, y, netwotk_outputs , epsilon):

        state = self.get_state(x, y)

        # Get valid actions
        mask = self.get_action_mask(x, y)

        valid_actions = np.where(mask)[0]

        # ---------------------------------------------
        # EXPLORATION
        # ---------------------------------------------

        if np.random.rand() < epsilon:

            action = np.random.choice(valid_actions)

        # ---------------------------------------------
        # EXPLOITATION
        # ---------------------------------------------

        else:

            best_index = np.argmax(netwotk_outputs)
            action = valid_actions[best_index]

        return action


    # ---------------------------------------------------------
    # GET NEXT STATE
    # ---------------------------------------------------------

    def get_next_state(self, x, y, action):

        if action == 0:          # Up

            next_x = x
            next_y = y - 1

        elif action == 1:        # Down

            next_x = x
            next_y = y + 1

        elif action == 2:        # Left

            next_x = x - 1
            next_y = y

        elif action == 3:        # Right

            next_x = x + 1
            next_y = y

        else:

            raise ValueError("Invalid action")

        return next_x, next_y

    def step(self, x, y, action):

        # Calculate next position
        next_x, next_y = self.get_next_state(
            x, y, action
        )

        return next_x, next_y