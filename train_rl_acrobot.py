import asyncio
import gymnasium as gym
from tamer.agent_rl_acrobot import Tamer

async def main():
    # Setup environment
    env = gym.make('Acrobot-v1', render_mode='rgb_array')

    # Hyperparameters
    discount_factor = 0.99
    epsilon = 0.1
    min_eps = 0
    num_episodes = 5
    tame = True
    tamer_training_timestep = 0.3

    # Initialize agent
    agent = Tamer(env, num_episodes, discount_factor, epsilon, min_eps, tame, tamer_training_timestep)

    print("Starting training phase")
    await agent.train(model_file_to_save_prefix='/Users/noahbayard/Downloads/TAMER-main/saved_rl_models_acrobot/autosave')

if __name__ == '__main__':
    asyncio.run(main())
