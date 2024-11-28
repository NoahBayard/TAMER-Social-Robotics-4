import asyncio
import gymnasium as gym
from tamer.agent_mountain_car import Tamer

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
    await agent.train(model_file_to_save_prefix='/Users/noahbayard/Downloads/TAMER-main/saved_models/autosave')

if __name__ == '__main__':
    asyncio.run(main())