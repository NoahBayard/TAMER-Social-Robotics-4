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

ACROBOT_ACTION_MAP = {0: 'left', 1: 'none', 2: 'right'}
MODELS_DIR = Path(__file__).parent.joinpath('saved_rl_models')
LOGS_DIR = Path(__file__).parent.joinpath('logs')

class SGDFunctionApproximator:
    """ SGD function approximator with RBF preprocessing. """
    def __init__(self, env):
        observation_examples = np.array(
            [env.observation_space.sample() for _ in range(10000)], dtype='float64'
        )
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)
        self.featurizer = pipeline.FeatureUnion(
            [
                ('rbf1', RBFSampler(gamma=5.0, n_components=100)),
                ('rbf2', RBFSampler(gamma=2.0, n_components=100)),
                ('rbf3', RBFSampler(gamma=1.0, n_components=100)),
                ('rbf4', RBFSampler(gamma=0.5, n_components=100)),
            ]
        )
        self.featurizer.fit(self.scaler.transform(observation_examples))
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate='constant')
            model.partial_fit([self.featurize_state(env.reset()[0])], [0])
            self.models.append(model)

    def predict(self, state, action=None):
        features = self.featurize_state(state)
        if action is None:
            return [m.predict([features])[0] for m in self.models]
        else:
            return self.models[action].predict([features])[0]

    def update(self, state, action, td_target):
        features = self.featurize_state(state)
        self.models[action].partial_fit([features], [td_target])

    def featurize_state(self, state):
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
        self.tame = tame
        self.ts_len = ts_len
        self.env = env
        self.uuid = uuid.uuid4()
        self.output_dir = output_dir
        self.discount_factor = discount_factor
        self.epsilon = epsilon if not tame else 0
        self.num_episodes = num_episodes
        self.min_eps = min_eps
        self.epsilon_step = (epsilon - min_eps) / num_episodes
        self.H = SGDFunctionApproximator(env)  # Human reinforcement model
        self.Q = SGDFunctionApproximator(env)  # Q-learning model
        self.reward_log_columns = ['Episode', 'Ep start ts', 'Feedback ts', 'Human Reward', 'Environment Reward']
        self.reward_log_path = os.path.join(self.output_dir, f'{self.uuid}.csv')

        if model_file_to_load is not None:
            print(f'Loaded pretrained model: {model_file_to_load}')
            self.load_model(filename=model_file_to_load)

    def act(self, state, alpha=0.8):
        """
        Choose an action using a combination of human feedback (H) and Q-learning (Q).
        The alpha parameter controls the weight of H during action selection.
        """
        if np.random.random() < 1 - self.epsilon:
            preds_H = np.array(self.H.predict(state))
            preds_Q = np.array(self.Q.predict(state))
            combined_preds = alpha * preds_H + (1 - alpha) * preds_Q
            return np.argmax(combined_preds)
        else:
            return np.random.randint(0, self.env.action_space.n)

    def _train_episode(self, episode_index, disp, alpha=0.8):
        """
        Train the agent for a single episode.
        Initially, human feedback (H) has a higher weight (controlled by alpha) to guide early learning.
        """
        print(f'Episode: {episode_index + 1}  Timestep:', end='')
        cv2.namedWindow('OpenAI Gymnasium Training', cv2.WINDOW_NORMAL)
        tot_reward = 0
        state, _ = self.env.reset()
        ep_start_time = dt.datetime.now().time()
        max_timesteps = 400
        timestep = 0
        with open(self.reward_log_path, 'a+', newline='') as write_obj:
            dict_writer = DictWriter(write_obj, fieldnames=self.reward_log_columns)
            dict_writer.writeheader()
            for ts in count():
                if timestep >= max_timesteps:
                    print(f'  Reached max timesteps: {max_timesteps}')
                    break
                timestep += 1
                print(f' {ts}', end='')
                frame_bgr = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
                cv2.imshow('OpenAI Gymnasium Training', frame_bgr)
                key = cv2.waitKey(25)
                if key == 27:
                    break
                action = self.act(state, alpha=alpha)
                if self.tame:
                    disp.show_action(action)
                next_state, reward, done, info, _ = self.env.step(action)
                if not self.tame:
                    td_target = reward + self.discount_factor * np.max(self.Q.predict(next_state)) if not done else reward
                    self.Q.update(state, action, td_target)
                else:
                    now = time.time()
                    while time.time() < now + self.ts_len:
                        time.sleep(0.01)
                        human_reward = disp.get_scalar_feedback()
                        feedback_ts = dt.datetime.now().time()
                        if human_reward != 0:
                            dict_writer.writerow({
                                'Episode': episode_index + 1,
                                'Ep start ts': ep_start_time,
                                'Feedback ts': feedback_ts,
                                'Human Reward': human_reward,
                                'Environment Reward': reward
                            })
                            self.H.update(state, action, human_reward)
                            # Update Q-values with human feedback to reinforce desired behavior
                            td_target = human_reward + self.discount_factor * np.max(self.Q.predict(next_state))
                            self.Q.update(state, action, td_target)
                            break
                tot_reward += reward
                if done:
                    print(f'  Reward: {tot_reward}')
                    cv2.destroyAllWindows()
                    break
                stdout.write('\b' * (len(str(ts)) + 1))
                state = next_state
        if self.epsilon > self.min_eps:
            self.epsilon -= self.epsilon_step
        # Reduce alpha gradually to let Q-learning take over more as training progresses
        alpha = max(0.1, alpha * 0.99)

    async def train(self, model_file_to_save_prefix=None):
        disp = None
        if self.tame:
            from .interface import Interface
            disp = Interface(action_map=ACROBOT_ACTION_MAP)
        alpha = 0.8  # Start with high influence of human feedback
        for i in range(self.num_episodes):
            self._train_episode(i, disp, alpha=alpha)
            if model_file_to_save_prefix:
                self.save_model(f"{model_file_to_save_prefix}_episode_{i + 1}")
        print('\nCleaning up...')
        self.env.close()

    def play(self, n_episodes=1, render=False):
        if render:
            cv2.namedWindow('OpenAI Gymnasium Playing', cv2.WINDOW_NORMAL)
        self.epsilon = 0
        ep_rewards = []
        max_timesteps = 400
        for i in range(n_episodes):
            state = self.env.reset()[0]
            done = False
            tot_reward = 0
            timestep = 0
            while not done:
                if timestep >= max_timesteps:
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
                        break
                state = next_state
            ep_rewards.append(tot_reward)
            print(f'Episode: {i + 1} Reward: {tot_reward}')
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
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'wb') as f:
            pickle.dump((self.H, self.Q), f)

    def load_model(self, filename):
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'rb') as f:
            self.H, self.Q = pickle.load(f)
