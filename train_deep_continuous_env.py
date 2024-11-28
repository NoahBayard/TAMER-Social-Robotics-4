import asyncio
import gymnasium as gym
from tamer.agent_deep_mc import Tamer
import os

async def main():
    # Setup environment
    env = gym.make('MountainCar-v0', render_mode='rgb_array')

    # Hyperparameters
    discount_factor = 1
    epsilon = 0
    min_eps = 0
    num_episodes = 5
    tame = True
    tamer_training_timestep = 0.3

    # Initialize agent
    agent = Tamer(env, num_episodes, discount_factor, epsilon, min_eps, tame, tamer_training_timestep)

    print("Starting training phase")
    save_dir = '/Users/noahbayard/Downloads/TAMER-main/saved_models_cartpole'
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

    for episode in range(num_episodes):
        # Generate the desired model filename
        model_file = os.path.join(save_dir, f"autosave_mc_deep_episode_{episode + 1}.p")
        print(f"Training and saving model for episode {episode + 1} to {model_file}")

        # Ensure the `Tamer` class saves exactly to this file
        await agent.train(model_file_to_save=model_file)

if __name__ == '__main__':
    asyncio.run(main())
