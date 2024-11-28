import pickle
import cv2
import numpy as np
import gymnasium as gym
import uuid
import datetime as dt
import os
from pathlib import Path
from csv import DictWriter
import time
from itertools import count

# Define the action map for Taxi environment
TAXI_ACTION_MAP = {
    0: 'south',
    1: 'north',
    2: 'east',
    3: 'west',
    4: 'pickup',
    5: 'dropoff'
}

# Directories for models and logs
MODELS_DIR = Path(__file__).parent.joinpath('saved_models')
LOGS_DIR = Path(__file__).parent.joinpath('logs')

class TabularQFunctionApproximator:
    """ Tabular function approximator adapted for discrete state spaces. """
    def __init__(self, num_states, num_actions):
        self.q_table = np.zeros((num_states, num_actions))

    def predict(self, state, action=None):
        if action is None:
            return self.q_table[state]
        else:
            return self.q_table[state][action]

    def update(self, state, action, td_target):
        self.q_table[state][action] = td_target
        # Print updated Q-table value for the state-action pair
        print(f"Updated Q-table for state {state}, action {action}: {self.q_table[state]}")

class Tamer():
    def __init__(
        self,
        env,
        num_episodes,
        discount_factor=1,
        epsilon=0.1,
        min_eps=0.01,
        tame=True,
        ts_len=0.2,
        output_dir=LOGS_DIR,
        model_file_to_load=None,
        seed=None
    ):
        self.tame = tame
        self.ts_len = ts_len
        self.env = env
        self.seed = seed  # Store the seed
        self.uuid = uuid.uuid4()
        self.output_dir = output_dir
        if model_file_to_load is not None:
            print(f'Loaded pretrained model: {model_file_to_load}')
            self.load_model(filename=model_file_to_load)
        else:
            num_states = env.observation_space.n
            num_actions = env.action_space.n
            if tame:
                self.H = TabularQFunctionApproximator(num_states, num_actions)
            else:
                self.Q = TabularQFunctionApproximator(num_states, num_actions)
        self.discount_factor = discount_factor
        self.epsilon = epsilon if not tame else 0
        self.num_episodes = num_episodes
        self.min_eps = min_eps
        self.epsilon_step = (epsilon - min_eps) / num_episodes
        self.reward_log_columns = [
            'Episode',
            'Ep start ts',
            'Feedback ts',
            'Human Reward',
            'Environment Reward',
        ]
        self.reward_log_path = os.path.join(self.output_dir, f'{self.uuid}.csv')
        self.last_q_table = None
        self.last_state = None

    def act(self, state):
        """ Epsilon-greedy Policy with small random noise to break symmetry """
        if np.random.random() < 1 - self.epsilon:
            preds = self.H.predict(state) if self.tame else self.Q.predict(state)
            # Adding a small random noise to the Q-values to avoid deterministic action selection
            noise = np.random.uniform(0, 1e-6, size=preds.shape)  # Small noise to break ties
            preds = preds + noise
            return np.argmax(preds)
        else:
            return np.random.randint(0, self.env.action_space.n)

    def _reset_environment(self):
        # Reset the environment using the provided seed if available
        if self.seed is not None:
            state, info = self.env.reset(seed=self.seed)
        else:
            state, info = self.env.reset()
        return state, info
    
    def _train_episode(self, episode_index, disp):
        print(f'Episode: {episode_index + 1}  Timestep:', end='')
        cv2.namedWindow('OpenAI Gymnasium Training', cv2.WINDOW_NORMAL)
        tot_reward = 0
        self.env = gym.make('Taxi-v3', render_mode='rgb_array')
        state, info = self._reset_environment()  # Use the new reset method
        state = self.state_to_index(state)
        print(f'Initial state: {info}')
        ep_start_time = dt.datetime.now().time()

        # Open the log file in append mode and write header only once
        with open(self.reward_log_path, 'a+', newline='') as write_obj:
            dict_writer = DictWriter(write_obj, fieldnames=self.reward_log_columns)

            # Write header only once per episode
            if episode_index == 0:
                dict_writer.writeheader()

            for ts in count():
                print(f' {ts}', end='')
                # Render the environment
                frame_bgr = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
                cv2.imshow('OpenAI Gymnasium Training', frame_bgr)
                key = cv2.waitKey(25)
                if key == 27:  # Exit on 'ESC'
                    break

                # Select action
                action = self.act(state)
                if self.tame:
                    disp.show_action(action)

                # Execute the action and get next state and reward
                next_state, reward, done, info, _ = self.env.step(action)
                next_state = self.state_to_index(next_state)

                if self.tame:
                    # Wait for human feedback for a specified duration (`ts_len`)
                    now = time.time()
                    human_reward = 0
                    #print("\nWaiting for human feedback...")

                    while time.time() < now + self.ts_len:
                        human_reward = disp.get_scalar_feedback()  # Get feedback from the interface
                        if human_reward != 0:  # If non-zero feedback is provided
                            feedback_ts = dt.datetime.now().time()
                            # Log the feedback
                            dict_writer.writerow({
                                'Episode': episode_index + 1,
                                'Ep start ts': ep_start_time,
                                'Feedback ts': feedback_ts,
                                'Human Reward': human_reward,
                                'Environment Reward': reward
                            })

                            # Update the Q-value with both environment reward and human feedback
                            current_q = self.H.predict(state, action)

                            # Combined reward
                            combined_reward = reward + human_reward

                            # Update the Q-table
                            self.H.update(state, action, combined_reward)
                            print(f'State transition: current state {state}, next state {next_state}, reward {combined_reward}')

                            break  # Exit loop once feedback is received
                else:
                    # Standard environment-based Q-learning update
                    td_target = reward if done else reward + self.discount_factor * np.max(self.Q.predict(next_state))
                    self.Q.update(state, action, td_target)

                tot_reward += reward
                state = next_state  # Transition to the next state

                if done:
                    print(f' Reward: {tot_reward}')
                    break

        cv2.destroyWindow('OpenAI Gymnasium Training')

        # Decay epsilon after the episode
        if self.epsilon > self.min_eps:
            self.epsilon -= self.epsilon_step

    async def train(self, model_file_to_save_prefix=None, seed_list=None):
        disp = None
        if self.tame:
            from .interface import Interface
            disp = Interface(action_map=TAXI_ACTION_MAP)

        # Use a list of seeds if provided, otherwise use the default seed
        if seed_list is None:
            seed_list = [self.seed] * self.num_episodes

        for i in range(self.num_episodes):
            self.seed = seed_list[i] if seed_list[i] is not None else self.seed  # Set seed for each episode
            self._train_episode(i, disp)

            if model_file_to_save_prefix:
                self.save_model(f"{model_file_to_save_prefix}_episode_{i + 1}")

        print('\nCleaning up...')
        self.env.close()

    def play(self, n_episodes=1, render=False):
        if render:
            cv2.namedWindow('OpenAI Gymnasium Playing', cv2.WINDOW_NORMAL)
        
        ep_rewards = []  # Initialize a list to store the rewards for each episode
        for i in range(n_episodes):
            state, info = self._reset_environment()  # Reset the environment
            state = self.state_to_index(state)
            done = False
            tot_reward = 0
            print(f"Starting Episode {i + 1}")

            while not done:
                action = self.act(state)
                next_state, reward, done, info, _ = self.env.step(action)
                next_state = self.state_to_index(next_state)
                tot_reward += reward

                if render:
                    frame_bgr = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
                    cv2.imshow('OpenAI Gymnasium Playing', frame_bgr)
                    key = cv2.waitKey(1)
                    if key == 27:  # Exit on 'ESC'
                        break

                state = next_state  # Transition to the next state
            ep_rewards.append(tot_reward)  # Append total reward for the episode
            print(f"Episode {i + 1} finished with total reward: {tot_reward}")
        
        if render:
            cv2.destroyWindow('OpenAI Gymnasium Playing')

        return ep_rewards  # Return the list of rewards


    def state_to_index(self, state):
        if isinstance(state, tuple):
            return state[0] * 1000 + state[1] * 100 + state[2] * 10 + state[3]
        return state

    def save_model(self, filename):
        print(f"Saving model to {filename}")
        if self.tame:
            with open(filename + '.p', 'wb') as file:
                pickle.dump(self.H, file)
        else:
            with open(filename + '.p', 'wb') as file:
                pickle.dump(self.Q, file)

    def load_model(self, filename):
        print(f"Loading model from {filename}")
        if not filename.endswith('.p'):
            filename += '.p'
        with open(filename, 'rb') as file:
            if self.tame:
                self.H = pickle.load(file)
            else:
                self.Q = pickle.load(file)

    def evaluate(self, n_episodes=100):
        """
        Evaluate the agent's performance by playing `n_episodes` without exploration (epsilon = 0).
        
        Args:
            n_episodes (int): Number of episodes to run during evaluation.
        """
        print('Evaluating agent')
        episode_times = []  # to store duration for each episode
        rewards = []

        # Play the episodes using the same method as before but without exploration
        rewards = self.play(n_episodes=n_episodes, render=False)

        # Calculate the metrics based on the rewards from the play method
        avg_reward = np.mean(rewards)
        avg_duration = np.mean([ep[3] for ep in episode_times])  # average duration per episode
        avg_timesteps = np.mean([ep[2] for ep in episode_times])  # average timesteps per episode

        print(
            f'\nAverage total episode reward over {n_episodes} episodes: {avg_reward:.2f}\n'
            f'Average duration per episode: {avg_duration:.2f} seconds\n'
            f'Average timesteps per episode: {avg_timesteps:.2f}'
        )

        return rewards, avg_reward, avg_duration, avg_timesteps
