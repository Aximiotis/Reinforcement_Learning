import TN
import Robot
import qn
import Map
import ReplayBuffer
import DQN_Agent
from torch import tensor
from torch import optim
import torch
import numpy as np


state_dim = 2
action_dim = 1
max_size = 1000

Buffer = ReplayBuffer.ReplayBuffer(
    max_size,
    state_dim,
    action_dim
)

width = 10
height = 10
MaxObstacles = 2

map_instance = Map.Map(
    width,
    height,
    MaxObstacles
)

grid = map_instance.get_map()
map_instance.set_target(9, 9)


agent = DQN_Agent.Agent(
    width * height,
    4,
    width,
    height
)

agent.set_learning_rates(
    alpha=0.1,
    gamma=0.9
)

robot = Robot.Robot(0, 0)


episodes = 1000
max_steps = 1000

QN = qn.QN(2, 100, 200, 4)
TN = TN.TN(2, 100, 200, 4)

weights1, weights2, weights3, bias1, bias2, bias3 = QN.get_weight()

TN.set_weight(
    weights1,
    weights2,
    weights3,
    bias1,
    bias2,
    bias3
)


learning_start = 50
train_frequency = 40
Tn_frequency = 200

batch_size = 64

learning_rate = 0.001
gamma = 0.9

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995

count = 0


optim_adam = optim.Adam(
    QN.parameters(),
    lr=learning_rate,
    weight_decay=1e-4
)


for i in range(episodes):

    robot.restart(0, 0)

    done = False
    steps = 0

    while not done and steps < max_steps:

        x, y = robot.get_position()

        # =========================================
        # CURRENT STATE
        # =========================================

        state = tensor(
            [x, y],
            dtype=torch.float32
        )

        # =========================================
        # Q-NETWORK
        # =========================================

        q_values = QN(state)

        q_values = q_values.detach().numpy()

        # =========================================
        # ACTION MASK
        # =========================================

        mask = np.array(
            agent.get_action_mask(x, y),
            dtype=bool
        )

        q_values[~mask] = -np.inf

        valid_actions = np.where(mask)[0]

        # =========================================
        # ε-GREEDY
        # =========================================

        if np.random.rand() < epsilon:

            action = np.random.choice(valid_actions)

        else:

            action = np.argmax(q_values)

        # =========================================
        # NEXT STATE
        # =========================================

        next_x, next_y = agent.get_next_state(
            x,
            y,
            action
        )

        # =========================================
        # REWARD
        # =========================================

        reward = -1
        done = False

        if grid[next_y][next_x] == -10:

            reward = -100
            done = True

        elif grid[next_y][next_x] == 100:

            reward = 100
            done = True

            print(
                f"Reached goal on episode {i}"
            )

        # =========================================
        # MOVE ROBOT
        # =========================================

        if not done or grid[next_y][next_x] == 100:

            robot.move(
                next_x=next_x,
                next_y=next_y
            )

        # =========================================
        # STORE EXPERIENCE
        # =========================================

        Buffer.add(
            [x, y],
            action,
            reward,
            [next_x, next_y],
            float(done)
        )

        count += 1

        # =========================================
        # TRAIN Q-NETWORK
        # =========================================

        if (
            count >= learning_start
            and count % train_frequency == 0
        ):

            indices = Buffer.sample(
                batch_size=batch_size
            )

            batch = Buffer.convert_to_array(
                indices=indices
            )

            states = torch.tensor(
                batch[:, 0:2],
                dtype=torch.float32
            )

            actions = torch.tensor(
                batch[:, 2],
                dtype=torch.long
            )

            rewards = torch.tensor(
                batch[:, 3],
                dtype=torch.float32
            )

            next_states = torch.tensor(
                batch[:, 4:6],
                dtype=torch.float32
            )

            dones = torch.tensor(
                batch[:, 6],
                dtype=torch.float32
            )

            # =====================================
            # TARGET NETWORK
            # =====================================

            with torch.no_grad():

                next_q_values = TN(next_states)

                # Mask invalid actions for next states
                next_masks = []

                for s in next_states:

                    nx = int(s[0].item())
                    ny = int(s[1].item())

                    next_masks.append(
                        agent.get_action_mask(nx, ny)
                    )

                next_masks = torch.tensor(
                    next_masks,
                    dtype=torch.bool
                )

                next_q_values[~next_masks] = -torch.inf

                max_next_q_values = torch.max(
                    next_q_values,
                    dim=1
                ).values

            # =====================================
            # TARGET
            # =====================================

            targets = rewards + gamma * (
                1 - dones
            ) * max_next_q_values

            # =====================================
            # Q-NETWORK
            # =====================================

            q_values = QN(states)

            current_q_values = q_values.gather(
                1,
                actions.unsqueeze(1)
            ).squeeze(1)

            # =====================================
            # LOSS
            # =====================================

            loss = torch.nn.functional.smooth_l1_loss(
                current_q_values,
                targets
            )

            # =====================================
            # UPDATE Q-NETWORK
            # =====================================

            optim_adam.zero_grad()

            loss.backward()

            optim_adam.step()

        # =========================================
        # UPDATE TARGET NETWORK
        # =========================================

        if count % Tn_frequency == 0:

            weights1, weights2, weights3, bias1, bias2, bias3 = QN.get_weight()

            TN.set_weight(
                weights1,
                weights2,
                weights3,
                bias1,
                bias2,
                bias3
            )

        steps += 1

    # =============================================
    # EPSILON DECAY
    # =============================================

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )