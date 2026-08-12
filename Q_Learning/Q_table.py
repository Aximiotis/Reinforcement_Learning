import numpy as np


class Q_table:

    def __init__(self, num_states, num_actions, grid_width, grid_height):

        self.states = num_states
        self.actions = num_actions

        # Q-table
        # Rows    -> states
        # Columns -> actions
        #
        # 0 = Up
        # 1 = Down
        # 2 = Left
        # 3 = Right
        self.table = np.zeros((num_states, num_actions))

        # Learning parameters
        self.learning_rate = 0.1      # alpha
        self.discount_rate = 0.9      # gamma

        # Grid dimensions
        self.grid_width = grid_width
        self.grid_height = grid_height


    # ---------------------------------------------------------
    # GET Q-TABLE
    # ---------------------------------------------------------

    def get_table(self):

        return self.table


    # ---------------------------------------------------------
    # SET LEARNING PARAMETERS
    # ---------------------------------------------------------

    def set_learning_rates(self, alpha, gamma):

        self.learning_rate = alpha
        self.discount_rate = gamma


    # ---------------------------------------------------------
    # STATE <-> (X,Y)
    # ---------------------------------------------------------

    def get_state(self, x, y):

        return y * self.grid_width + x


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

    def choose_action(self, x, y, epsilon):

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

            # Q values only for valid actions
            q_values = self.table[state, valid_actions]

            # Best valid action
            best_index = np.argmax(q_values)

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


    # ---------------------------------------------------------
    # Q-LEARNING UPDATE
    # ---------------------------------------------------------

    def update(
        self,
        x,
        y,
        action,
        reward,
        next_x,
        next_y,
        done
    ):

        state = self.get_state(x, y)

        current_q = self.table[state, action]

        # Terminal state
        if done:

            target = reward

        else:

            next_state = self.get_state(
                next_x,
                next_y
            )

            next_mask = self.get_action_mask(
                next_x,
                next_y
            )

            valid_next_actions = np.where(
                next_mask
            )[0]

            max_next_q = np.max(
                self.table[
                    next_state,
                valid_next_actions
                ]
            )

            target = (
                reward
                + self.discount_rate * max_next_q
            )

        # Q-learning update
        new_q = current_q + self.learning_rate * (
            target - current_q
     )

        self.table[state, action] = new_q
    # ---------------------------------------------------------
    # COMPLETE STEP
    # ---------------------------------------------------------
    def step(self, x, y, action):

        # Calculate next position
        next_x, next_y = self.get_next_state(
            x, y, action
        )

        return next_x, next_y