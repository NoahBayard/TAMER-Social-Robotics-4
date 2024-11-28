import os
import numpy as np
import matplotlib.pyplot as plt
from tamer.agent_rl_cartpole import Tamer
import gymnasium as gym
import pickle

MODELS_DIR = '/Users/noahbayard/Downloads/TAMER-main/saved_rl_models_cartpole/'

def evaluate_models(env, models_dir, num_episodes=10):
    avg_rewards = []
    model_files = sorted([f for f in os.listdir(models_dir) if f.endswith('.p')])
    
    print("Model files found:", model_files)  # Debugging output

    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        print(f'Evaluating model: {model_path}')  # Debugging output
        
        # Load the model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Create an agent and assign the loaded model to it
        agent = Tamer(env, num_episodes, tame=True)
        agent.H, agent.Q = model  # Assign both the H and Q models

        # Evaluate the agent
        try:
            avg_reward = agent.evaluate(n_episodes=num_episodes)
        except Exception as e:
            print(f"Error while evaluating model {model_file}: {e}")
            continue  # Skip to the next model if there is an error

        avg_rewards.append((model_file, avg_reward))

    return avg_rewards

def plot_average_rewards(avg_rewards):
    if not avg_rewards:
        print("No models were evaluated. Ensure model files exist and can be loaded.")
        return

    models = [model for model, _ in avg_rewards]
    rewards = [reward for _, reward in avg_rewards]

    plt.plot(models, rewards, marker='o')
    plt.xlabel('Model File')
    plt.ylabel('Average Reward')
    plt.xticks(rotation=45, ha='right')
    plt.title('Average Reward for Each Trained Model')
    plt.tight_layout()
    plt.show()

def main():
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    avg_rewards = evaluate_models(env, MODELS_DIR, num_episodes=20)
    plot_average_rewards(avg_rewards)

if __name__ == '__main__':
    main()
