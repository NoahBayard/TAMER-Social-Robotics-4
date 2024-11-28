import os
import numpy as np
import matplotlib.pyplot as plt
from tamer.player_rl_taxi import Tamer
import gymnasium as gym
import pickle

MODELS_DIR = '/Users/noahbayard/Downloads/TAMER-main/saved_rl_models_taxi'

# Function to evaluate models with a given seed list
def evaluate_models(env, models_dir, num_episodes, seed_list):
    avg_rewards = []
    model_files = sorted([f for f in os.listdir(models_dir) if f.endswith('.p')])

    print("Model files found:", model_files)  # Debugging output

    # Iterate through each model file and its corresponding seed from seed_list
    for model_file, seed in zip(model_files, seed_list):
        model_path = os.path.join(models_dir, model_file)
        print(f'Evaluating model: {model_path}')  # Debugging output
        
        # Load the model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Create an agent and assign the loaded model to it with the given seed
        agent = Tamer(env, num_episodes, tame=False, seed=seed)  # Set `tame=False` for TAMER+RL evaluation
        agent.Q = model  # Set the loaded model to Q (reinforcement learning model)

        # Evaluate the agent and calculate the average reward for the model
        try:
            avg_reward = agent.evaluate(n_episodes=num_episodes)
        except Exception as e:
            print(f"Error while evaluating model {model_file}: {e}")
            continue  # Skip to the next model if there is an error
        
        # Append the model name and the average reward to the list
        avg_rewards.append((model_file, avg_reward))

    return avg_rewards

# Function to plot average rewards for models
def plot_average_rewards(avg_rewards):
    if not avg_rewards:
        print("No models were evaluated. Ensure model files exist and can be loaded.")
        return

    models = [model for model, _ in avg_rewards]  # Extract model file names
    rewards = [reward for _, reward in avg_rewards]  # Extract average rewards

    # Create the plot
    plt.plot(models, rewards, marker='o')
    plt.xlabel('Model File')
    plt.ylabel('Average Reward per 100 episodes')
    plt.xticks(rotation=45, ha='right')
    plt.title('Average Reward for Each Trained Model')
    plt.tight_layout()

    # Save the plot as an image (e.g., PNG)
    plt.savefig('average_rewards_plot_taxi_tamer_rl.png')
    print("Plot saved as 'average_rewards_plot_taxi_tamer_rl.png'")

    # Optionally, show the plot if possible
    try:
        plt.show()  # Attempt to show the plot if the environment supports it
    except Exception as e:
        print(f"Error displaying plot: {e}")

# Main function
def main():
    env = gym.make('Taxi-v3', render_mode='rgb_array')
    seed_list = [1, 15, 43, 14, 45]  # Seeds 43
    avg_rewards = evaluate_models(env, MODELS_DIR, num_episodes=100, seed_list=seed_list)
    plot_average_rewards(avg_rewards)

if __name__ == '__main__':
    main()
