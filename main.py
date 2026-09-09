import numpy as np
import matplotlib.pyplot as plt

maze = np.array([
    [0, 1, 1, 1, 1],
    [0, 0, 0, 1, 1],
    [0, 1, 0, 1 ,1],
    [0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0]
])

num_episodes = 5000

alpha = 0.1
gamma = 0.9
epsilon = 0.5

wall_reward = -10
goal_reward = 50
step_reward = -1

start = (0, 0)
goal = (4, 4)

actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

Q = np.zeros(maze.shape + (len(actions), ))

def is_valid(pos):
    r, c = pos
    if r < 0 or r >= maze.shape[0]:
        return False
    if c < 0 or  c >= maze.shape[1]:
        return False
    if maze[r, c] == 1:
        return False
    return True

def choose_action(state):
    if np.random.random() < epsilon:
        return np.random.randint(len(actions))
    else:
        return np.argmax(Q[state])

rewards_all_episodes = []
for episode in range(num_episodes):
    state = start
    total_reward = 0
    done = False

    while not done:
        action_index = choose_action(state)
        action = actions[action_index]

        next_state = (state[0] + action[0], state[1] + action[1])

        if not is_valid(next_state):
            reward = wall_reward
            done = True
        elif next_state == goal:
            reward = goal_reward
            done = True
        else:
            reward = step_reward

        old_value = Q[state][action_index]
        next_max = np.max(Q[next_state]) if is_valid(next_state) else 0

        Q[state][action_index] = old_value + alpha * (reward + gamma * next_max - old_value)

        state = next_state
        total_reward += reward

    epsilon = max(0.01, epsilon * 0.995)
    rewards_all_episodes.append(total_reward)

def heat_map(Q):
    best_q = np.max(Q, axis=2)

    plt.figure(figsize=(6, 6))
    plt.imshow(best_q, cmap='viridis')
    plt.colorbar(label='Max Q value')
    plt.title('Best Q-values per state')

    for r in range(maze.shape[0]):
        for c in range(maze.shape[1]):
            if maze[r, c] == 1:
                text = "WALL"
            else:
                text = f"{best_q[r, c]:.1f}"
            plt.text(c, r, text, ha='center', va='center', color='white', fontsize=9)

    plt.xticks(range(maze.shape[1]))
    plt.yticks(range(maze.shape[0]))
    plt.show()
heat_map(Q)