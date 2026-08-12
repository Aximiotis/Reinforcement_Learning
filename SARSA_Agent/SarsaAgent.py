import numpy as np


class SarsaAgent:

    def __init__(
        self,
        num_states,
        num_actions,
        grid_width,
        grid_height
    ):

        self.states = num_states
        self.actions = num_actions

        # =================================================
        # Q-TABLE
        # =================================================
        #
        # 0 = Up
        # 1 = Down
        # 2 = Left
        # 3 = Right
        #

        self.table = np.zeros(
            (num_states, num_actions)
        )

        # =================================================
        # LEARNING PARAMETERS
        # =================================================

        self.learning_rate = 0.1
        self.discount_rate = 0.9

        # =================================================
        # GRID DIMENSIONS
        # =================================================

        self.grid_width = grid_width
        self.grid_height = grid_height


    # =====================================================
    # GET Q-TABLE
    # =====================================================

    def get_table(self):

        return self.table


    # =====================================================
    # SET LEARNING PARAMETERS
    # =====================================================

    def set_learning_rates(
        self,
        alpha,
        gamma
    ):

        self.learning_rate = alpha
        self.discount_rate = gamma


    # =====================================================
    # STATE -> INDEX
    # =====================================================

    def get_state(
        self,
        x,
        y
    ):

        return (
            y * self.grid_width
            + x
        )


    # =====================================================
    # INDEX -> STATE
    # =====================================================

    def get_coordinates(
        self,
        state
    ):

        y = state // self.grid_width

        x = state % self.grid_width

        return x, y


    # =====================================================
    # ACTION MASK
    # =====================================================
    #
    # 0 = Up
    # 1 = Down
    # 2 = Left
    # 3 = Right
    #
    # True  = valid
    # False = invalid
    #

    def get_action_mask(
        self,
        x,
        y
    ):

        mask = [
            True,   # Up
            True,   # Down
            True,   # Left
            True    # Right
        ]

        # ---------------------------------------------
        # UP
        # ---------------------------------------------

        if y == 0:

            mask[0] = False

        # ---------------------------------------------
        # DOWN
        # ---------------------------------------------

        if y == self.grid_height - 1:

            mask[1] = False

        # ---------------------------------------------
        # LEFT
        # ---------------------------------------------

        if x == 0:

            mask[2] = False

        # ---------------------------------------------
        # RIGHT
        # ---------------------------------------------

        if x == self.grid_width - 1:

            mask[3] = False

        return mask


    # =====================================================
    # CHOOSE ACTION
    # =====================================================

    def choose_action(
        self,
        x,
        y,
        epsilon
    ):

        # ---------------------------------------------
        # CURRENT STATE
        # ---------------------------------------------

        state = self.get_state(
            x,
            y
        )

        # ---------------------------------------------
        # ACTION MASK
        # ---------------------------------------------

        mask = self.get_action_mask(
            x,
            y
        )

        # ---------------------------------------------
        # VALID ACTIONS
        # ---------------------------------------------

        valid_actions = np.where(mask)[0]

        # =============================================
        # EXPLORATION
        # =============================================

        if np.random.rand() < epsilon:

            action = np.random.choice(
                valid_actions
            )

        # =============================================
        # EXPLOITATION
        # =============================================

        else:

            q_values = self.table[
                state,
                valid_actions
            ]

            best_index = np.argmax(
                q_values
            )

            action = valid_actions[
                best_index
            ]

        return action


    # =====================================================
    # GET NEXT STATE
    # =====================================================

    def get_next_state(
        self,
        x,
        y,
        action
    ):

        # ---------------------------------------------
        # UP
        # ---------------------------------------------

        if action == 0:

            next_x = x
            next_y = y - 1

        # ---------------------------------------------
        # DOWN
        # ---------------------------------------------

        elif action == 1:

            next_x = x
            next_y = y + 1

        # ---------------------------------------------
        # LEFT
        # ---------------------------------------------

        elif action == 2:

            next_x = x - 1
            next_y = y

        # ---------------------------------------------
        # RIGHT
        # ---------------------------------------------

        elif action == 3:

            next_x = x + 1
            next_y = y

        else:

            raise ValueError(
                "Invalid action"
            )

        return next_x, next_y


    # =====================================================
    # SARSA UPDATE
    # =====================================================

    def update(
        self,
        x,
        y,
        action,
        reward,
        next_x,
        next_y,
        next_action,
        done
    ):

        # ---------------------------------------------
        # CURRENT STATE
        # ---------------------------------------------

        state = self.get_state(
            x,
            y
        )

        # ---------------------------------------------
        # CURRENT Q-VALUE
        #
        # Q(s,a)
        # ---------------------------------------------

        current_q = self.table[
            state,
            action
        ]

        # =============================================
        # TERMINAL STATE
        # =============================================

        if done:

            target = reward

        # =============================================
        # NON-TERMINAL STATE
        # =============================================

        else:

            # -----------------------------------------
            # NEXT STATE
            # -----------------------------------------

            next_state = self.get_state(
                next_x,
                next_y
            )

            # -----------------------------------------
            # SARSA
            #
            # Q(s',a')
            # -----------------------------------------

            next_q = self.table[
                next_state,
                next_action
            ]

            # -----------------------------------------
            # SARSA TARGET
            # -----------------------------------------

            target = (
                reward
                + self.discount_rate * next_q
            )

        # =============================================
        # SARSA UPDATE
        # =============================================

        new_q = current_q + self.learning_rate * (
            target - current_q
        )

        # ---------------------------------------------
        # UPDATE Q TABLE
        # ---------------------------------------------

        self.table[
            state,
            action
        ] = new_q


    # =====================================================
    # STEP
    # =====================================================

    def step(
        self,
        x,
        y,
        action
    ):

        next_x, next_y = self.get_next_state(
            x,
            y,
            action
        )

        return next_x, next_y