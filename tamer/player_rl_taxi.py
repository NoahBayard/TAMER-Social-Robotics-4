import os
import cv2
import numpy as np
import gymnasium as gym
from tamer.agent_taxi import Tamer

def main(saved_models_dir, render=True, seed_list=None):
    """
    Load all trained TAMER+RL models from the saved_models directory and play one episode for each model using given seeds.

    Args:
        saved_models_dir: Path to the directory containing saved model files.
        render: Whether to render the environment.
        seed_list: List of seeds to be used for resetting the environment.
    """
    # List all model files in the saved_models directory
    model_files = [f for f in os.listdir(saved_models_dir) if f.startswith("autosave_episode_") and f.endswith(".p")]

    if not model_files:
        print("No autosave models found in the directory.")
        return

    print(f"Found {len(model_files)} models. Playing one episode per model.")

    # Sort the model files to ensure they are processed in the correct order
    model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Sort by episode number

    # If seed list is provided, make sure its length matches the number of models
    if seed_list is None:
        seed_list = [1] * len(model_files)

    # Loop through each model file and corresponding seed
    for model_file, seed in zip(model_files, seed_list):
        model_path = os.path.join(saved_models_dir, model_file)
        print(f"Loading model from {model_path} with seed: {seed}")

        # Create and reset environment with the seed
        environment = gym.make('Taxi-v3', render_mode='rgb_array')
        state, info = environment.reset(seed=seed)  # Set the seed here

        # Load the trained model using `Tamer` class
        tamer_agent = Tamer(environment, num_episodes=1, tame=False, seed=seed)  # Pass the seed to the agent, set `tame=False` for RL
        tamer_agent.load_model(model_path)

        # Play one episode for the current model
        rewards = tamer_agent.play(n_episodes=1, render=render)

        # Print episode rewards
        if rewards is not None:
            print(f"Model: {model_file} - Episode 1: Total reward = {rewards[0]}")
        else:
            print(f"No rewards returned from model {model_file}.")

if __name__ == "__main__":
    # Directory with saved models
    saved_models_dir = '/Users/noahbayard/Downloads/TAMER-main/saved_rl_models_taxi'

    # Define the seed list that matches the training seeds
    seed_list = [1, 15, 43, 14, 45]  # Replace with the actual seed values used during training

    main(saved_models_dir=saved_models_dir, render=True, seed_list=seed_list)
