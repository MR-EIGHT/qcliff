import numpy as np
from environment import CliffEnvironment
from collections import defaultdict
import matplotlib.pyplot as plt


def greedy(Q, state):
    return np.argmax(Q[state])


def epsilon_greedy(Q, state, nA, epsilon=0.1):
    best_action = np.argmax(Q[state])
    action = np.ones(nA, dtype=np.float32) * epsilon / nA
    action[best_action] += 1 - epsilon
    return action


def plot(x, y, labels):
    size = len(x)
    plt.plot([x[i] for i in range(size) if i % 10 == 0], [y[i] for i in range(size) if i % 10 == 0], label=labels)


def print_policy(Q):
    env = CliffEnvironment()
    result = ""
    for k in range(env.height):
        line = ""
        for j in range(env.width):
            action = np.argmax(Q[(j, k)])
            if action == 0:
                line += "↑ "
            elif action == 1:
                line += "↓ "
            elif action == 2:
                line += "← "
            else:
                line += "→ "
        result = line + "\n" + result
    print(result)


def SARSA(env=CliffEnvironment(), episode_nums=1000, learning_rate=0.3, discount_factor=.95, epsilon=0.1,
          epsilon_decay=0.00005):
    Q = defaultdict(lambda: np.zeros(env.nA))
    sarsa_rewards = []
    for _ in range(episode_nums):

        if epsilon > 0.01:
            epsilon -= epsilon_decay

        env.reset()
        state, done = env.observation()
        action = epsilon_greedy(Q, state, env.nA, epsilon)
        probs = action
        action = np.random.choice(np.arange(env.nA), p=probs)
        sum_reward = 0.0

        while not done:
            next_state, reward, done = env.step(action)

            next_action = epsilon_greedy(Q, next_state, env.nA, epsilon)
            probs = next_action
            next_action = np.random.choice(np.arange(env.nA), p=probs)
            Q[state][action] = Q[state][action] + learning_rate * (
                    reward + discount_factor * Q[next_state][next_action] - Q[state][action])
            state = next_state
            action = next_action
            sum_reward += reward
            if done:
                sarsa_rewards.append(sum_reward)

    return Q, sarsa_rewards


def Q_learning(env=CliffEnvironment(), learning_rate=0.3, episode_number=1000, discount_factor=.95, epsilon=0.1,
               epsilon_decay=0.00005):
    Q = defaultdict(lambda: np.zeros(env.nA))
    ql_rewards = []
    for _ in range(episode_number):
        env.reset()
        if epsilon > 0.01:
            epsilon -= epsilon_decay
        cur_state, done = env.observation()
        sum_reward = 0.0

        while not done:
            prob = epsilon_greedy(Q, cur_state, env.nA, epsilon)
            action = np.random.choice(np.arange(env.nA), p=prob)
            next_state, reward, done = env.step(action)
            next_action = greedy(Q, next_state)
            Q[cur_state][action] = Q[cur_state][action] + learning_rate * (
                    reward + discount_factor * Q[next_state][next_action] - Q[cur_state][action])
            cur_state = next_state
            sum_reward += reward
            if done:
                ql_rewards.append(sum_reward)

    return Q, ql_rewards


def ql_driver():
    print('SARSA')
    env = CliffEnvironment()
    Q, rewards = SARSA(env)
    average_rewards = []
    for i in range(10):
        Q, rewards = SARSA(env)
        average_rewards = np.array(rewards) if len(average_rewards) == 0 else average_rewards + np.array(rewards)

    average_rewards = average_rewards / 10.0
    plot(range(1000), average_rewards, labels='SARSA')
    print_policy(Q)


def sarsa_driver():
    print('Q-Learning')
    average_rewards2 = []
    for i in range(10):
        Q, rewards = Q_learning()
        average_rewards2 = np.array(rewards) if len(average_rewards2) == 0 else average_rewards2 + np.array(rewards)

    average_rewards2 = average_rewards2 / 10.0
    plot(range(1000), average_rewards2, labels='Q-LEARNING')
    print_policy(Q)


if __name__ == '__main__':
    ql_driver()
    sarsa_driver()

    plt.title('Q-Learning / SARSA')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.ylim(-150, 0)
    plt.legend()
    plt.show()
