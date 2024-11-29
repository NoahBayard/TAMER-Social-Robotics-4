#Requires futher implementation


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import uuid
import datetime as dt
import cv2
import os
from pathlib import Path
from itertools import count
from csv import DictWriter
import time
import gymnasium as gym
import random

import pickle

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

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.index = 0

    def push(self, state, action, human_feedback, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, human_feedback, next_state))
        else:
            self.buffer[self.index] = (state, action, human_feedback, next_state)
            self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


class DeepFunctionApproximator(nn.Module):
    def __init__(self, input_dim, action_space):
        super(DeepFunctionApproximator, self).__init__()
        self.action_space = action_space
        self.discount_factor = 0
        
        # Define separate output layers for each action
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Separate output layers for each action
        self.output_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(128, 64),  # Action-specific hidden layer
            nn.ReLU(),
            nn.Linear(64, 1)  # Output layer
        ) for _ in range(action_space)])  # Separate layers for each action

        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)
        self.criterion = nn.MSELoss()

    def predict(self, state):
        """Predict feedback for all actions for a given state."""
        state_tensor = self._one_hot_encode(state)
        
        # Pass through the shared layers
        shared_output = self.shared_layer(state_tensor)
        
        # Get predictions for each action, using its own specific layers
        action_values = [output_layer(shared_output) for output_layer in self.output_layers]
        
        # Return all action values (one per action)
        return torch.cat(action_values, dim=1).squeeze()

    def _one_hot_encode(self, state):
        """One-hot encode the state."""
        state_tensor = torch.zeros(500).float()  # 500 for Taxi-v3 state space
        state_tensor[int(state)] = 1  # Set the state index to 1 (one-hot encoding)
        return state_tensor.unsqueeze(0)  # Add batch dimension

    def update(self, state, action, human_reward):
        # Prepare the state tensor
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_index = int(action)

        # Get the model's prediction for the current state
        predictions = self.predict(state_tensor)
        target = predictions.clone().detach()

        # Human feedback influence
        human_feedback_factor = 10  # This can be tuned, adjust as necessary
        # ONLY use human feedback for the target
        target[action_index] = human_feedback_factor * human_reward

        # Compute the loss (Mean Squared Error) and update the model
        loss = self.criterion(predictions, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



class Tamer():
    def __init__(
        self,
        env,
        num_episodes,
        discount_factor=1,
        epsilon=0.0,
        min_eps=0.0,
        tame=True,
        ts_len=0.2,
        output_dir=LOGS_DIR,
        model_file_to_load=None,
        seed=None,
        buffer_capacity=10000,  # Capacity for the replay buffer
        batch_size=32          # Size of the batch for experience replay
    ):
        self.tame = tame
        self.ts_len = ts_len
        self.env = env
        self.seed = seed  # Store the seed
        self.uuid = uuid.uuid4()
        self.output_dir = output_dir
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size

        # Initialize the replay buffer
        self.replay_buffer = ReplayBuffer(capacity=self.buffer_capacity)

        if model_file_to_load is not None:
            print(f'Loaded pretrained model: {model_file_to_load}')
            self.load_model(filename=model_file_to_load)
        else:
            if tame:
                input_dim = env.observation_space.n  # For Taxi-v3, this should be 500
                self.H = DeepFunctionApproximator(
                    input_dim=input_dim,  # Set to 500 for Taxi-v3
                    action_space=env.action_space.n
                )
            else:
                self.Q = DeepFunctionApproximator(
                    input_dim=env.observation_space.n,  # Assuming a discrete space
                    action_space=env.action_space.n
                )
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
            # Detach the predictions from the computation graph before adding noise
            preds = preds.detach()  # Detach to stop tracking gradients
            # Adding a small random noise to the Q-values to avoid deterministic action selection

            action = np.argmax(preds)
        else:
            action = np.random.randint(0, self.env.action_space.n)
        
        # Convert action to a scalar (if it is a tensor) to avoid KeyError in show_action
        action = action.item() if isinstance(action, torch.Tensor) else action

        return action

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
        print(f"State before conversion: {state} (type: {type(state)})")
        feedback_scaling_factor = 1

        # Open the log file in append mode and write header only once
        with open(self.reward_log_path, 'a+', newline='') as write_obj:
            dict_writer = DictWriter(write_obj, fieldnames=self.reward_log_columns)

            # Write header only once (on first episode)
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
                                'Feedback ts': feedback_ts,
                                'Human Reward': human_reward,
                                'Environment Reward': reward
                            })
                            print(f"State {state} received human feedback: {human_reward}")

                            # Show Q-values after human feedback
                            preds_after_feedback = self.H.predict(state) if self.tame else self.Q.predict(state)
                            print(f"Q-values after feedback for state {state}: {preds_after_feedback.tolist()}")

                            # Add experience to the replay buffer
                            self.replay_buffer.push(state, action, human_reward, next_state)

                            # Update the H-function (Deep TAMER)
                            self.H.update(state, action, human_reward)
                            break  # Exit loop once feedback is received
                else:
                    # Standard environment-based Q-learning update
                    td_target = reward if done else reward + self.discount_factor * np.max(self.Q.predict(next_state))
                    self.Q.update(state, action, td_target)

                tot_reward += reward
                state = next_state  # Transition to the next state

                # Sample a batch from the replay buffer and update the model
                if self.replay_buffer.size() > self.batch_size:
                    batch = self.replay_buffer.sample(self.batch_size)
                    for state, action, human_reward, next_state in batch:
                        self.H.update(state, action, human_reward)

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
            self._train_episode(i, disp)  # Train for the current episode

            if model_file_to_save_prefix:
                # Save the model after the episode if requested
                self.save_model(f"{model_file_to_save_prefix}_episode_{i + 1}")

        print('\nCleaning up...')
        self.env.close()  # Clean up after training


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
    
    
