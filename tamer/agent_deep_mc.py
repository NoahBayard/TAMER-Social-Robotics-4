import datetime as dt
import os
import pickle
import time
import uuid
from itertools import count
from pathlib import Path
from sys import stdout
from csv import DictWriter
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler


MOUNTAINCAR_ACTION_MAP = {0: 'left', 1: 'none', 2: 'right'}

MODELS_DIR = Path(__file__).parent.joinpath('saved_deep_models')
LOGS_DIR = Path(__file__).parent.joinpath('logs')

class DeepFunctionApproximator(nn.Module):
    def __init__(self, input_dim, action_space):
        super(DeepFunctionApproximator, self).__init__()
        self.action_space = action_space
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_space)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

    def predict(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
        return self.model(state_tensor).detach().numpy().squeeze()

    def update(self, state, action, human_reward):
        self.optimizer.zero_grad()
        state_tensor = torch.tensor(state, dtype=torch.float32)
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension

        action_index = int(action)
        reward_tensor = torch.tensor(human_reward, dtype=torch.float32)

        predictions = self.model(state_tensor)
        target = predictions.clone().detach()
        target[0, action_index] = reward_tensor

        loss = self.criterion(predictions, target)
        loss.backward()
        self.optimizer.step()


class Tamer:
    def __init__(
        self,
        env,
        num_episodes,
        discount_factor=1,
        epsilon=0,
        min_eps=0,
        tame=True,
        ts_len=0.2,
        output_dir=LOGS_DIR,
        model_file_to_load=None
    ):
        self.num_episodes = num_episodes
        self.tame = tame
        self.ts_len = ts_len
        self.env = env
        self.uuid = uuid.uuid4()
        self.output_dir = output_dir
        self.scaler = StandardScaler()
        print(f"Initializing Tamer Agent in {'TAMER' if tame else 'DQN'} mode.")

        # Initialize epsilon and related attributes
        self.epsilon = epsilon
        self.min_eps = min_eps
        self.epsilon_step = (epsilon - min_eps) / num_episodes if num_episodes > 0 else 0

        # Fit the scaler using sampled states from the environment
        print("Fitting state scaler...")
        sampled_states = np.array([env.observation_space.sample() for _ in range(1000)])
        self.scaler.fit(sampled_states)
        print("State scaler fitted.")

        if model_file_to_load is not None:
            print(f'Loaded pretrained model: {model_file_to_load}')
            self.load_model(filename=model_file_to_load)
        else:
            if tame:
                self.H = DeepFunctionApproximator(
                    input_dim=env.observation_space.shape[0],
                    action_space=env.action_space.n
                )
            else:
                self.Q = DeepFunctionApproximator(
                    input_dim=env.observation_space.shape[0],
                    action_space=env.action_space.n
                )


        # Initialize reward log attributes
        self.reward_log_columns = [
            'Episode',
            'Ep start ts',
            'Feedback ts',
            'Human Reward',
            'Environment Reward',
        ]
        self.reward_log_path = os.path.join(self.output_dir, f'{self.uuid}.csv')

    def act(self, state):
        state = np.array(state).reshape(1, -1)  # Ensure state is 2D
        state = self.scaler.transform(state)  # Normalize the state
        if self.tame:
            h_values = self.H.predict(state)
            return np.argmax(h_values)
        else:
            q_values = self.Q.predict(state)
            return np.argmax(q_values)



    def _train_episode(self, episode_index, disp):
        print(f"Starting Episode {episode_index + 1}")
        cv2.namedWindow('OpenAI Gymnasium Training', cv2.WINDOW_NORMAL)
        tot_reward = 0
        state, _ = self.env.reset()
        state = np.array(state).reshape(1, -1)  # Ensure state is 2D
        state = self.scaler.transform(state)  # Normalize the initial state
        print(f"Normalized Initial State: {state}")
        ep_start_time = dt.datetime.now().time()

        with open(self.reward_log_path, 'a+', newline='') as write_obj:
            dict_writer = DictWriter(write_obj, fieldnames=self.reward_log_columns)
            if episode_index == 0:
                dict_writer.writeheader()

            for ts in count():
                frame_bgr = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
                cv2.imshow('OpenAI Gymnasium Training', frame_bgr)
                key = cv2.waitKey(25)
                if key == 27:
                    print("Exiting due to user interruption.")
                    break

                action = self.act(state)
                if self.tame:
                    disp.show_action(action)

                next_state, reward, done, info, _ = self.env.step(action)
                next_state = np.array(next_state).reshape(1, -1)  # Ensure next_state is 2D
                next_state = self.scaler.transform(next_state)  # Normalize the next state

                if self.tame:
                    now = time.time()
                    while time.time() < now + self.ts_len:
                        time.sleep(0.01)
                        human_reward = disp.get_scalar_feedback()
                        if human_reward != 0:
                            feedback_ts = dt.datetime.now().time()
                            print(f"Human feedback: {human_reward} at Timestep {ts}")
                            dict_writer.writerow({
                                'Episode': episode_index + 1,
                                'Ep start ts': ep_start_time,
                                'Feedback ts': feedback_ts,
                                'Human Reward': human_reward,
                                'Environment Reward': reward
                            })
                            self.H.update(state, action, human_reward)
                            break

                tot_reward += reward

                if done:
                    print(f"Episode {episode_index + 1} completed. Total reward: {tot_reward}")
                    cv2.destroyAllWindows()
                    break

                state = next_state
        if self.epsilon > self.min_eps:
            self.epsilon -= self.epsilon_step
            print(f"Epsilon decreased to: {self.epsilon}")

    async def train(self, model_file_to_save=None):
        """
        TAMER (or Q-learning) training loop.
        Args:
            model_file_to_save: base name for saving models (without extension).
        """
        disp = None
        if self.tame:
            from .interface import Interface
            disp = Interface(action_map=MOUNTAINCAR_ACTION_MAP)

        # Resolve model file base path to the correct directory
        if model_file_to_save:
            base_path = Path(model_file_to_save)
            if not base_path.is_absolute():
                base_path = MODELS_DIR / base_path.name

        for i in range(self.num_episodes):
            print(f"Num episode: {i + 1}/{self.num_episodes}")
            self._train_episode(i, disp)

            # Save model after each episode with a unique filename
            if model_file_to_save is not None:
                episode_filename = base_path.with_name(f"{base_path.stem}_episode_{i + 1}.pth")
                self.save_model(filename=str(episode_filename))

        print('\nCleaning up...')
        self.env.close()
        
    def play(self, n_episodes=1, render=False, max_timesteps=500):
        """
        Play episodes with the trained model.

        Args:
            n_episodes: Number of episodes to play.
            render: Whether to render the environment.
            max_timesteps: Maximum number of timesteps per episode to avoid getting stuck.
        """
        if render:
            cv2.namedWindow('OpenAI Gymnasium Playing', cv2.WINDOW_NORMAL)
        self.epsilon = 0  # No exploration during evaluation
        ep_rewards = []
        for i in range(n_episodes):
            state = self.env.reset()[0]
            state = self.scaler.transform([state])  # Normalize the state
            done = False
            tot_reward = 0
            timestep = 0  # Track the number of timesteps

            while not done and timestep < max_timesteps:
                action = self.act(state)
                next_state, reward, done, info, _ = self.env.step(action)
                next_state = self.scaler.transform([next_state])  # Normalize the next state
                tot_reward += reward
                timestep += 1  # Increment timestep count

                if render:
                    frame_bgr = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
                    cv2.imshow('OpenAI Gymnasium Playing', frame_bgr)
                    key = cv2.waitKey(25)
                    if key == 27:  # Exit on ESC key
                        break

                state = next_state

            print(f'Episode: {i + 1} Reward: {tot_reward} Timesteps: {timestep}')
            ep_rewards.append(tot_reward)

        self.env.close()
        if render:
            cv2.destroyAllWindows()
        return ep_rewards

    def evaluate(self, n_episodes=100):
        print('Evaluating agent')
        rewards = self.play(n_episodes=n_episodes)
        avg_reward = np.mean(rewards)
        print(f'Average total episode reward over {n_episodes} episodes: {avg_reward:.2f}')
        return avg_reward

    def save_model(self, filename):
        """
        Save the model's state dictionary to the specified filename.
        """
        model = self.H if self.tame else self.Q
        filename = filename + '.pth' if not filename.endswith('.pth') else filename
        save_path = MODELS_DIR.joinpath(filename)
        torch.save(model.state_dict(), save_path)  # Save only the state dictionary
        print(f"Model state_dict saved to {save_path}")

    def load_model(self, filename):
        """
        Load the model's state dictionary from the specified filename.
        """
        model = self.H if self.tame else self.Q
        filename = filename + '.pth' if not filename.endswith('.pth') else filename
        load_path = MODELS_DIR.joinpath(filename)
        model.load_state_dict(torch.load(load_path, weights_only=True))  # Load only the state dictionary
        model.eval()  # Set model to evaluation mode
        print(f"Model state_dict loaded from {load_path}")
