import Map
import Robot
import SarsaAgent


if __name__ == "__main__":

    # =====================================================
    # ENVIRONMENT PARAMETERS
    # =====================================================

    width = 10
    height = 10

    MaxObstacles = 20

    # =====================================================
    # CREATE MAP
    # =====================================================

    map_instance = Map.Map(
        width,
        height,
        MaxObstacles
    )

    grid = map_instance.get_map()

    # Set target
    map_instance.set_target(9, 9)

    print("Initial Map:")
    print(map_instance.get_ascii_map())

    # =====================================================
    # SARSA AGENT
    # =====================================================

    agent = SarsaAgent.SarsaAgent(
        width * height,
        4,
        width,
        height
    )

    agent.set_learning_rates(
        alpha=0.1,
        gamma=0.9
    )

    # =====================================================
    # ROBOT
    # =====================================================

    robot = Robot.Robot(0, 0)

    # =====================================================
    # RL PARAMETERS
    # =====================================================

    episodes = 1000
    max_steps = 100

    epsilon = 1.0
    epsilon_min = 0.001
    epsilon_decay = 0.995

    # =====================================================
    # TRAINING
    # =====================================================

    for episode in range(episodes):

        # ---------------------------------------------
        # RESET ROBOT
        # ---------------------------------------------

        robot.restart(0, 0)

        done = False

        # ---------------------------------------------
        # INITIAL STATE
        # ---------------------------------------------

        x, y = robot.get_position()

        # ---------------------------------------------
        # INITIAL ACTION
        # ---------------------------------------------

        action = agent.choose_action(
            x,
            y,
            epsilon
        )

        # ---------------------------------------------
        # EPISODE
        # ---------------------------------------------

        for step in range(max_steps):

            # =========================================
            # 1. CURRENT STATE
            # =========================================

            x, y = robot.get_position()

            # =========================================
            # 2. TAKE ACTION
            # =========================================

            next_x, next_y = agent.get_next_state(
                x,
                y,
                action
            )

            # =========================================
            # 3. ENVIRONMENT
            # =========================================

            # Check grid boundaries first

            if not (
                0 <= next_x < width
                and
                0 <= next_y < height
            ):

                # Out of bounds

                reward = -100
                done = True

            elif grid[next_y][next_x] == -10:

                # Collision with obstacle

                reward = -100
                done = True

            elif grid[next_y][next_x] == 100:

                # Goal reached

                reward = 100
                done = True

            else:

                # Normal movement

                reward = -1
                done = False

            # =========================================
            # 4. CHOOSE NEXT ACTION a'
            # =========================================

            if not done:

                next_action = agent.choose_action(
                    next_x,
                    next_y,
                    epsilon
                )

            else:

                next_action = None

            # =========================================
            # 5. SARSA UPDATE
            # =========================================

            agent.update(
                x,
                y,
                action,
                reward,
                next_x,
                next_y,
                next_action,
                done
            )

            # =========================================
            # 6. MOVE ROBOT
            # =========================================

            if not done:

                robot.move(
                    next_x,
                    next_y
                )

                # -------------------------------------
                # SARSA:
                #
                # a' becomes the current action
                #
                # -------------------------------------

                action = next_action

            # =========================================
            # 7. TERMINAL STATE
            # =========================================

            if done:

                if reward == 100:

                    print(
                        f"Episode {episode + 1}: "
                        f"Goal reached in "
                        f"{step + 1} steps"
                    )

                break

        # =============================================
        # 8. EPSILON DECAY
        # =============================================

        epsilon = max(
            epsilon_min,
            epsilon * epsilon_decay
        )

    # =====================================================
    # TRAINING FINISHED
    # =====================================================

    print("\nTraining finished.")

    print(
        f"Final epsilon: {epsilon:.4f}"
    )

    # =====================================================
    # FINAL MAP
    # =====================================================

    print("\nFinal Map:")

    print(
        map_instance.get_ascii_map()
    )