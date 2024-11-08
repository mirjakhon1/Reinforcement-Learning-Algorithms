import numpy as np


"""
This script implements the Value Iteration algorithm for solving a Gridworld problem in Reinforcement Learning.
It calculates the optimal state-value function for the Gridworld, to find the optimal policy.
"""


def value_iteration(grid_size=10, discount_factor=0.9, theta=1e-4, max_iterations=1000):
    """
    Performs Value Iteration for a Gridworld environment.

    Parameters:
        grid_size (int): Size of the grid (grid_size x grid_size).
        discount_factor (float): Discount factor for future rewards.
        theta (float): Small threshold for determining convergence.
        max_iterations (int): Maximum number of iterations to perform.

    Returns:
        np.ndarray: The optimal state-value function.
    """

    # Create a grid world matrix
    value_function = np.zeros((grid_size, grid_size)).astype(np.float32) 
    
    new_value_function = np.copy(value_function)
    
    # Actions: up, down, left, right
    actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # Set the goal state (bottom-right corner)
    goal_state = (grid_size - 1, grid_size - 1)


    for iteration in range(max_iterations):
        delta = 0  # Initialize delta for convergence check

        # Perform policy evalution, which uses old policy (value_function), in-place value function update
        # Loop over all states in the grid
        for x in range(grid_size):
            for y in range(grid_size):
                if (x, y) == goal_state:
                    continue  # Skip the goal state
                

                action_values = []
                for action in actions:
                    new_x, new_y = x + action[0], y + action[1]
                    
                    # Check for boundaries
                    if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
                        next_state = (new_x, new_y)
                    else:
                        # If out of bounds, stay in the same state
                        next_state = (x, y)
                    
                    # Reward of -1 for each step
                    reward = -1

                    # Calculate value function for state s
                    action_value = reward + discount_factor * value_function[next_state]
                    action_values.append(action_value)

                # Update the value function with the maximum action value (greedy method)
                best_action_value = max(action_values)
                delta = max(delta, abs(best_action_value - value_function[x, y]))
                value_function[x, y] = best_action_value
        
        # Check for convergence
        if delta < theta:
            print(f"Value iteration converged after {iteration + 1} iterations.")
            break
    else:
        print(f"Value iteration did not converge after {max_iterations} iterations.")
        
    # Return or process the value function further
    print("Optimal State-Value Function:")
    print(np.round(value_function, 3))
    return value_function



if __name__ == '__main__':
    value_iteration()
