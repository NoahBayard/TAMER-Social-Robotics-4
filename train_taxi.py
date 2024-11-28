import asyncio
import gymnasium as gym
from tamer.agent_taxi import Tamer  # Assuming the agent is named accordingly for the Taxi environment

async def main():
    # Setup environment
    env = gym.make('Taxi-v3', render_mode='rgb_array')

    # Hyperparameters
    discount_factor = 1
    epsilon = 0
    min_eps = 0
    num_episodes = 5  # Set to the number of seeds we will use
    tame = True
    tamer_training_timestep = 1.5

    # List of seeds to use for training
    seed_list = [1, 15, 43, 14, 45]  # Seeds 43

    # Initialize agent
    agent = Tamer(env, num_episodes, discount_factor, epsilon, min_eps, tame, tamer_training_timestep, seed=None)

    # Train agent using multiple seeds
    print("Starting training phase with multiple seeds")
    await agent.train(model_file_to_save_prefix='/home/linux/TAMER/TAMER-main/tamer/saved_models/autosave', seed_list=seed_list)

if __name__ == '__main__':
    asyncio.run(main())
