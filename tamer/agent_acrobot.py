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
from sklearn import pipeline, preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
import cv2

ACROBOT_ACTION_MAP = {0: 'torque left', 1: 'none', 2: 'torque right'}
MODELS_DIR = Path(__file__).parent.joinpath('saved_models_acrobot')
LOGS_DIR = Path(__file__).parent.joinpath('logs')

class SGDFunctionApproximator:
    """ SGD function approximator with RBF preprocessing. """
    def __init__(self, env):
        # Generate samples from the observation space to initialize the scaler
        observation_examples = np.array(
            [env.observation_space.sample() for _ in range(10000)], dtype='float64'
        )
        # Standardize features by removing the mean and scaling to unit variance
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)
        # Use RBF kernels to create feature representations of states
        self.featurizer = pipeline.FeatureUnion(
            [
                ('rbf1', RBFSampler(gamma=5.0, n_components=100)),
                ('rbf2', RBFSampler(gamma=2.0, n_components=100)),
                ('rbf3', RBFSampler(gamma=1.0, n_components=100)),
                ('rbf4', RBFSampler(gamma=0.5, n_components=100)),
            ]
        )
        self.featurizer.fit(self.scaler.transform(observation_examples))
        # Initialize an SGDRegressor for each action
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate='constant')
            # Fit the model with an initial state to initialize it
            model.partial_fit([self.featurize_state(env.reset()[0])], [0])
            self.models.append(model)

    def predict(self, state, action=None):
        # Predict the Q-values for a given state
        features = self.featurize_state(state)
        if action is None:
            return [m.predict([features])[0] for m in self.models]
        else:
            return self.models[action].predict([features])[0]

    def update(self, state, action, td_target):
        # Update the model for the given action using the target value
        features = self.featurize_state(state)
        self.models[action].partial_fit([features], [td_target])

    def featurize_state(self, state):
        # Transform the state into a feature representation using the scaler and featurizer
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]

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
        # Initialize TAMER or Q-learning agent
        self.tame = tame
        self.ts_len = ts_len  # Length of time to wait for human feedback
        self.env = env
        self.uuid = uuid.uuid4()  # Unique identifier for logging
        self.output_dir = output_dir
        if model_file_to_load is not None:
            # Load a pretrained model if specified
            print(f'Loaded pretrained model: {model_file_to_load}')
            self.load_model(filename=model_file_to_load)
        else:
            # Initialize a new model for TAMER or Q-learning
            if tame:
                self.H = SGDFunctionApproximator(env)
            else:
                self.Q = SGDFunctionApproximator(env)
        self.discount_factor = discount_factor
        self.epsilon = epsilon if not tame else 0  # Exploration rate for Q-learning
        self.num_episodes = num_episodes
        self.min_eps = min_eps
        self.epsilon_step = (epsilon - min_eps) / num_episodes  # Epsilon decay per episode
        # Columns for logging human and environment rewards
        self.reward_log_columns = [
            'Episode',
            'Ep start ts',
            'Feedback ts',
            'Human Reward',
            'Environment Reward',
        ]
        # Path to log file
        self.reward_log_path = os.path.join(self.output_dir, f'{self.uuid}.csv')

    def act(self, state):
        # Choose an action using epsilon-greedy policy
        if np.random.random() < 1 - self.epsilon:
            preds = self.H.predict(state) if self.tame else self.Q.predict(state)
            return np.argmax(preds)
        else:
            return np.random.randint(0, self.env.action_space.n)

    def _train_episode(self, episode_index, disp):
        # Train the agent for a single episode
        print(f'Episode: {episode_index + 1}  Timestep:', end='')
        cv2.namedWindow('OpenAI Gymnasium Training', cv2.WINDOW_NORMAL)
        tot_reward = 0
        state, _ = self.env.reset()
        ep_start_time = dt.datetime.now().time()
        max_timesteps = 400  # Limit the number of timesteps per episode to avoid long evaluations
        timestep = 0
        with open(self.reward_log_path, 'a+', newline='') as write_obj:
            dict_writer = DictWriter(write_obj, fieldnames=self.reward_log_columns)
            dict_writer.writeheader()
            for ts in count():
                if timestep >= max_timesteps:
                    # Stop if max timesteps is reached
                    print(f'  Reached max timesteps: {max_timesteps}')
                    break
                timestep += 1
                print(f' {ts}', end='')
                frame_bgr = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
                cv2.imshow('OpenAI Gymnasium Training', frame_bgr)
                key = cv2.waitKey(25)
                if key == 27:
                    # Exit if 'Esc' key is pressed
                    break
                action = self.act(state)
                if self.tame:
                    disp.show_action(action)
                next_state, reward, done, info, _ = self.env.step(action)
                if not self.tame:
                    # Update Q-values for Q-learning using temporal difference target
                    td_target = reward if done and next_state[0] >= 0.5 else reward + self.discount_factor * np.max(self.Q.predict(next_state))
                    self.Q.update(state, action, td_target)
                else:
                    # Wait for human feedback in TAMER
                    now = time.time()
                    while time.time() < now + self.ts_len:
                        time.sleep(0.01)
                        human_reward = disp.get_scalar_feedback()
                        feedback_ts = dt.datetime.now().time()
                        if human_reward != 0:
                            # Log human feedback and update model
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
                    # End the episode if the goal is reached
                    print(f'  Reward: {tot_reward}')
                    cv2.destroyAllWindows()
                    break
                stdout.write('\b' * (len(str(ts)) + 1))
                state = next_state
        # Decay epsilon after each episode
        if self.epsilon > self.min_eps:
            self.epsilon -= self.epsilon_step

    async def train(self, model_file_to_save_prefix=None):
        # Train the agent over multiple episodes
        disp = None
        if self.tame:
            from .interface import Interface
            disp = Interface(action_map=ACROBOT_ACTION_MAP)
        for i in range(self.num_episodes):
            self._train_episode(i, disp)
            # Save the model after each episode
            if model_file_to_save_prefix:
                self.save_model(f"{model_file_to_save_prefix}_episode_{i + 1}")
        print('\nCleaning up...')
        self.env.close()

    def play(self, n_episodes=1, render=False):
        # Play using the trained model
        if render:            
            cv2.namedWindow('OpenAI Gymnasium Playing', cv2.WINDOW_NORMAL)
        self.epsilon = 0  # Set epsilon to 0 to avoid exploration
        ep_rewards = []
        max_timesteps = 400  # Limit the number of timesteps per episode to avoid long evaluations
        for i in range(n_episodes):
            state = self.env.reset()[0]
            done = False
            tot_reward = 0
            timestep = 0
            while not done:
                if timestep >= max_timesteps:
                    # Stop if max timesteps is reached
                    print(f'Episode: {i + 1} Reached max timesteps: {max_timesteps}')
                    break
                timestep += 1
                action = self.act(state)
                next_state, reward, done, info, _ = self.env.step(action)
                tot_reward += reward
                if render:
                    frame_bgr = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
                    cv2.imshow('OpenAI Gymnasium Playing', frame_bgr)
                    key = cv2.waitKey(25)
                    if key == 27:
                        # Exit if 'Esc' key is pressed
                        break
                state = next_state
            ep_rewards.append(tot_reward)
            print(f'Episode: {i + 1} Reward: {tot_reward}')
        self.env.close()
        if render:
            cv2.destroyAllWindows()
        return ep_rewards

    def evaluate(self, n_episodes=100):
        # Evaluate the trained model over multiple episodes
        print('Evaluating agent')
        rewards = self.play(n_episodes=n_episodes)
        avg_reward = np.mean(rewards)
        print(f'Average total episode reward over {n_episodes} episodes: {avg_reward:.2f}')
        return avg_reward

    def save_model(self, filename):
        # Save the model to the specified file
        model = self.H if self.tame else self.Q
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, filename):
        # Load the model from the specified file
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'rb') as f:
            model = pickle.load(f)
        if self.tame:
            self.H = model
        else:
            self.Q = model