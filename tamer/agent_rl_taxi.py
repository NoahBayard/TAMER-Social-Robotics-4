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
import random

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
MODELS_DIR = Path(__file__).parent.joinpath('saved_rl_taxi_models')
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
        seed=None,
        alpha=0.5  # Learning rate to control the influence of human feedback vs Q-learning
    ):
        self.tame = tame
        self.ts_len = ts_len
        self.env = env
        self.seed = seed  # Store the seed
        self.uuid = uuid.uuid4()
        self.output_dir = output_dir
        self.alpha = alpha  # Alpha parameter to control the influence of human feedback
        if model_file_to_load is not None:
            print(f'Loaded pretrained model: {model_file_to_load}')
            self.load_model(filename=model_file_to_load)
        else:
            num_states = env.observation_space.n
            num_actions = env.action_space.n
            # Initialize both human feedback model (H) and Q-learning model (Q)
            self.H = TabularQFunctionApproximator(num_states, num_actions)
            self.Q = TabularQFunctionApproximator(num_states, num_actions)
        self.discount_factor = discount_factor
        self.epsilon = epsilon
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
            # Weighted combination of human feedback (H) and Q-learning (Q) values
            h_preds = self.H.predict(state)
            q_preds = self.Q.predict(state)
            combined_preds = self.alpha * h_preds + (1 - self.alpha) * q_preds
            
            # Adding a small random noise to the Q-values to avoid deterministic action selection
            noise = np.random.uniform(0, 1e-6, size=combined_preds.shape)
            combined_preds = combined_preds + noise
            return np.argmax(combined_preds)
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
                            current_q = self.Q.predict(state, action)
                            # Combined reward
                            combined_reward = reward + human_reward
                            # Temporal Difference target for Q-learning with human feedback
                            td_target = (1 - self.alpha) * current_q + self.alpha * combined_reward
                            self.Q.update(state, action, td_target)

                            # Update H with human feedback directly
                            self.H.update(state, action, human_reward)
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


    def state_to_index(self, state):
        if isinstance(state, tuple):
            return state[0] * 1000 + state[1] * 100 + state[2] * 10 + state[3]
        return state

    def evaluate(self, n_episodes=100, seed_list=None):
        """
        Evaluate the agent's performance by playing `n_episodes` with random seed selection from `seed_list`.
        
        Args:
            n_episodes (int): Number of episodes to run during evaluation.
            seed_list (list): List of seeds to randomize across episodes.
        """
        print('Evaluating agent')
        rewards = []
        
        for i in range(n_episodes):
            # Randomly select a seed for each episode if seed_list is provided
            if seed_list:
                self.seed = random.choice(seed_list)
            
            state, info = self._reset_environment()  # Reset the environment with the new seed
            state = self.state_to_index(state)
            done = False
            tot_reward = 0
            print(f"Starting Episode {i + 1} with Seed: {self.seed}")

            while not done:
                action = self.act(state)
                next_state, reward, done, info, _ = self.env.step(action)
                next_state = self.state_to_index(next_state)
                tot_reward += reward
                state = next_state  # Transition to the next state

            rewards.append(tot_reward)  # Append total reward for the episode
            print(f"Episode {i + 1} finished with total reward: {tot_reward}")
        
        # Calculate the average reward
        avg_reward = np.mean(rewards)
        print(f'\nAverage total episode reward over {n_episodes} episodes: {avg_reward:.2f}')
        return avg_reward


    def save_model(self, filename):
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'wb') as f:
            print(f"Saving model to {filename}")
            pickle.dump((self.H, self.Q), f)  # Save both H and Q as a tuple
            print("Model saved successfully!")


    def load_model(self, filename):
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'rb') as f:
            print(f"Loading model from {filename}")
            saved_data = pickle.load(f)
            
            # Ensure the saved data is a tuple of two TabularQFunctionApproximator objects
            if isinstance(saved_data, tuple) and len(saved_data) == 2:
                self.H, self.Q = saved_data
                print("Model loaded successfully!")
            else:
                raise ValueError("Loaded model file does not contain the expected tuple (H, Q).")


    def play(self, n_episodes=1, render=False):
        if render:
            cv2.namedWindow('OpenAI Gymnasium Playing', cv2.WINDOW_NORMAL)
        
        ep_rewards = []  # Initialize a list to store the rewards for each episode
        for i in range(n_episodes):
            state, info = self._reset_environment()  # Reset the environment
            print(self.seed)
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
