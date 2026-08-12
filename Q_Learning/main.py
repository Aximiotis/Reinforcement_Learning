import Map
import Q_table
import Robot
import numpy as np


if __name__ == "__main__":

    width = 10
    height = 10

    MaxObstacles = 20

    # -----------------------------
    # MAP
    # -----------------------------

    map_instance = Map.Map(
        width,
        height,
        MaxObstacles
    )

    grid = map_instance.get_map()

    map_instance.set_target(9, 9)

    print(map_instance.get_ascii_map())

    # -----------------------------
    # Q TABLE
    # -----------------------------

    q_table = Q_table.Q_table(
        width * height,
        4,
        width,
        height
    )

    q_table.set_learning_rates(
        alpha=0.1,
        gamma=0.9
    )

    # -----------------------------
    # ROBOT
    # -----------------------------

    robot = Robot.Robot(0, 0)

    # -----------------------------
    # RL PARAMETERS
    # -----------------------------

    episodes = 1000
    max_steps = 100

    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995

    # -----------------------------
    # TRAINING
    # -----------------------------

    for episode in range(episodes):

        robot.restart(0, 0)

        done = False

        for step in range(max_steps):

            # Current position
            x, y = robot.get_position()

            # --------------------------------
            # 1. CHOOSE ACTION
            # --------------------------------

            action = q_table.choose_action(
                x,
                y,
                epsilon
            )

            # --------------------------------
            # 2. GET NEXT STATE
            # --------------------------------

            next_x, next_y = q_table.get_next_state(
                x,
                y,
                action
            )

            # --------------------------------
            # 3. ENVIRONMENT
            # --------------------------------

            if grid[next_y][next_x] == -10:

                # Collision
                reward = -100
                done = True

            elif grid[next_y][next_x] == 100:

                # Goal
                reward = 100
                done = True

            else:

                # Normal movement
                reward = -1
                done = False

            # --------------------------------
            # 4. Q UPDATE
            # --------------------------------

            q_table.update(
                x,
                y,
                action,
                reward,
                next_x,
                next_y,
                done
            )

            # --------------------------------
            # 5. MOVE ROBOT
            # --------------------------------

            if not done:

                robot.move(
                    next_x,
                    next_y
                )

            # --------------------------------
            # 6. TERMINAL STATE
            # --------------------------------

            if done:

                if reward == 100:

                    print(
                        f"Episode {episode + 1}: "
                        f"Goal reached in {step + 1} steps"
                    )

                break

        # --------------------------------
        # EPSILON DECAY
        # --------------------------------

        epsilon = max(
            epsilon_min,
            epsilon * epsilon_decay
        )

    # -----------------------------
    # FINAL MAP
    # -----------------------------

    print("\nFinal map:")
    print(map_instance.get_ascii_map())