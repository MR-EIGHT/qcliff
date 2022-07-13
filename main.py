import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import turtle as t

environment_rows = 4
environment_columns = 12
q_values = np.zeros((environment_rows, environment_columns, 4))

actions = ['UP', 'RIGHT', 'DOWN', 'LEFT']

rewards = np.full((environment_rows, environment_columns), -1.)
rewards[3, 1:-1] = -100.
rewards[3, -1] = 100.

for row in rewards:
    print(row)


# function that determines if the specified location is a terminal state
def is_terminal_state(current_row_index, current_column_index):
    if rewards[current_row_index, current_column_index] == -1.:
        return False
    return True


# an epsilon greedy algorithm that will choose which action to take next (i.e., where to move next)
def get_next_action(current_row_index, current_column_index, epsilon):
    # if a randomly chosen value between 0 and 1 is less than epsilon,
    # then choose the most promising value from the Q-table for this state.
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_row_index, current_column_index])
    else:  # choose a random action
        return np.random.randint(4)


# define a function that will get the next location based on the chosen action
def get_next_location(current_row_index, current_column_index, action_index):
    new_row_index = current_row_index
    new_column_index = current_column_index
    if actions[action_index] == 'UP' and current_row_index > 0:
        new_row_index -= 1
    elif actions[action_index] == 'RIGHT' and current_column_index < environment_columns - 1:
        new_column_index += 1
    elif actions[action_index] == 'DOWN' and current_row_index < environment_rows - 1:
        new_row_index += 1
    elif actions[action_index] == 'LEFT' and current_column_index > 0:
        new_column_index -= 1
    if new_row_index == 3 and 0 < new_column_index < 11:
        new_row_index = 3
        new_column_index = 0
    return new_row_index, new_column_index


# Define a function that will get the shortest path between any location within the warehouse that
# the robot is allowed to travel and the item packaging location.
def get_shortest_path(start_row_index, start_column_index):
    current_row_index, current_column_index = start_row_index, start_column_index
    shortest_path = [[current_row_index, current_column_index]]
    # continue moving along the path until we reach the goal (i.e., the item packaging location)
    while not is_terminal_state(current_row_index, current_column_index):
        # get the best action to take
        action_index = get_next_action(current_row_index, current_column_index, .1)
        # move to the next location on the path, and add the new location to the list
        current_row_index, current_column_index = get_next_location(current_row_index, current_column_index,
                                                                    action_index)
        shortest_path.append([current_row_index, current_column_index])
    return shortest_path


# define training parameters
epsilon = 0.1  # the percentage of time when we should take the best action (instead of a random action)
discount_factor = .95  # discount factor for future rewards
learning_rate = 0.01  # the rate at which the agent should learn
eps_decay = 0.00005
# run through 1000 training episodes
for episode in range(1000):
    # get the starting location for this episode
    row_index, column_index = (3, 0)
    # continue taking actions (i.e., moving) until we reach a terminal state
    # (i.e., until we reach the item packaging area or crash into an item storage location)

    if epsilon > 0.01:
        epsilon -= eps_decay

    while not is_terminal_state(row_index, column_index):
        # choose which action to take (i.e., where to move next)
        action_index = get_next_action(row_index, column_index, epsilon)
        # perform the chosen action, and transition to the next state (i.e., move to the next location)
        old_row_index, old_column_index = row_index, column_index  # store the old row and column indexes
        row_index, column_index = get_next_location(row_index, column_index, action_index)
        # receive the reward for moving to the new state, and calculate the temporal difference
        reward = rewards[row_index, column_index]
        old_q_value = q_values[old_row_index, old_column_index, action_index]
        temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value
        # update the Q-value for the previous state and action pair
        new_q_value = old_q_value + (learning_rate * temporal_difference)
        q_values[old_row_index, old_column_index, action_index] = new_q_value
print('Training complete!')

print(get_shortest_path(3, 0))



