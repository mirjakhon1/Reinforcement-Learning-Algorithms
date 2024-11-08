import random
import numpy as np
import matplotlib.pyplot as plt

"""
This script simulates the multi-armed bandit problem using the epsilon-greedy algorithm.
It compares the estimated win rates of each machine with their actual win rates and visualizes the results.
"""

def generate_machine_means(num_machines=10, seed=22):
    """
    Generates the true mean rewards for each machine.

    Parameters:
        num_machines (int): The number of slot machines (arms).
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: An array of true mean rewards for each machine.
    """
    np.random.seed(seed)
    machine_means = np.random.normal(0, 1, num_machines)
    return machine_means

def test_all_machines(machine_means, test_cycles=1000):
    """
    Tests each machine by simulating plays and calculates the empirical mean reward.

    Parameters:
        machine_means (np.ndarray): The true mean rewards of the machines.
        test_cycles (int): The number of times to play each machine.

    Returns:
        None
    """
    num_machines = len(machine_means)
    for i in range(num_machines):
        print("************")
        print(f"Machine Nr. {i + 1}")
        true_mean = machine_means[i]
        rewards = np.random.normal(true_mean, 1, test_cycles)
        empirical_mean = np.mean(rewards)
        print(f"Calculated mean value: {empirical_mean:.4f}")
        print(f"Theoretical mean value: {true_mean:.4f}")

def display_machine_winrates(machine_means):
    """
    Displays a histogram of the true win rates of the machines.

    Parameters:
        machine_means (np.ndarray): The true mean rewards of the machines.

    Returns:
        None
    """
    plt.hist(machine_means, bins=len(machine_means), alpha=0.7, color='blue', edgecolor='black')
    plt.title('Win Rates of Machines')
    plt.xlabel('Win Rate')
    plt.ylabel('Frequency')
    plt.show()

def play_bandit(machine_means, num_iterations=10000, epsilon=0.1, reward_stddev=1.0):
    """
    Simulates playing the multi-armed bandit using the epsilon-greedy algorithm.

    Parameters:
        machine_means (np.ndarray): The true mean rewards of the machines.
        num_iterations (int): The number of times to play a machine.
        epsilon (float): The probability of exploring (choosing a random machine).
        reward_stddev (float): The standard deviation of the reward distribution.

    Returns:
        None
    """
    num_machines = len(machine_means)
    estimated_means = np.zeros(num_machines, dtype=np.float64)
    machine_draws = np.zeros(num_machines, dtype=np.uint64)
    total_reward = 0.0
    total_optimal_reward = 0.0

    for _ in range(num_iterations):
        if random.random() < epsilon:
            # Explore: choose a random machine
            machine_idx = random.randint(0, num_machines - 1)
        else:
            # Exploit: choose the machine with the highest estimated mean reward
            machine_idx = np.argmax(estimated_means)

        # Simulate reward from the chosen machine
        reward = np.random.normal(machine_means[machine_idx], reward_stddev)
        total_reward += reward

        # Update estimated mean reward for the chosen machine
        machine_draws[machine_idx] += 1
        n = machine_draws[machine_idx]
        estimated_means[machine_idx] += (reward - estimated_means[machine_idx]) / n

        # Calculate the optimal reward that could have been obtained
        optimal_reward = np.max(np.random.normal(machine_means, reward_stddev))
        total_optimal_reward += optimal_reward

    # Calculate performance score
    score = (total_reward / total_optimal_reward) * 100  # in percentage

    # Plotting results
    plt.figure(figsize=(10, 8))

    # Subplot 1: Actual win rates
    plt.subplot(4, 1, 1)
    plt.bar(range(num_machines), machine_means, color='yellow')
    plt.xlabel("Machine Index")
    plt.ylabel("Actual Mean Reward")
    plt.title("Actual Mean Rewards of Each Machine")

    # Subplot 2: Estimated win rates
    plt.subplot(4, 1, 2)
    plt.bar(range(num_machines), estimated_means, color='skyblue')
    plt.xlabel("Machine Index")
    plt.ylabel("Estimated Mean Reward")
    plt.title("Estimated Mean Rewards of Each Machine")

    # Subplot 3: Number of draws
    plt.subplot(4, 1, 3)
    plt.bar(range(num_machines), machine_draws, color='salmon')
    plt.xlabel("Machine Index")
    plt.ylabel("Number of Draws")
    plt.title("Number of Draws for Each Machine")

    # Subplot 4: Summary statistics
    plt.subplot(4, 1, 4)
    plt.text(0.1, 0.8, f"Total Reward: {total_reward:.2f}", fontsize=12)
    plt.text(0.1, 0.6, f"Total Optimal Reward: {total_optimal_reward:.2f}", fontsize=12)
    plt.text(0.1, 0.4, f"Performance Score: {score:.2f}%", fontsize=12)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Parameters
    NUM_MACHINES = 10
    NUM_ITERATIONS = 10000
    EPSILON = 0.1
    REWARD_STDDEV = 1.0

    # Generate true machine means
    machine_means = generate_machine_means(num_machines=NUM_MACHINES)

    # Uncomment to test all machines
    # test_all_machines(machine_means)

    # Uncomment to display machine win rates
    # display_machine_winrates(machine_means)

    # Play the bandit game
    play_bandit(
        machine_means=machine_means,
        num_iterations=NUM_ITERATIONS,
        epsilon=EPSILON,
        reward_stddev=REWARD_STDDEV
    )
