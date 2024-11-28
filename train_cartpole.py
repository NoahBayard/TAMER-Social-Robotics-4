import asyncio
import gymnasium as gym
from tamer.agent_cartpole import Tamer

async def main():
    # Setup environment
    env = gym.make('CartPole-v1', render_mode='rgb_array')

    # Hyperparameters
    discount_factor = 1
    epsilon = 0
    min_eps = 0
    num_episodes = 5
    tame = True
    tamer_training_timestep = 0.5

    # Initialize agent
    agent = Tamer(env, num_episodes, discount_factor, epsilon, min_eps, tame, tamer_training_timestep)

    print("Starting training phase")
    await agent.train(model_file_to_save_prefix='/Users/noahbayard/Downloads/TAMER-main/saved_models_cartpole/autosave')

if __name__ == '__main__':
    asyncio.run(main())