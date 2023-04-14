from abc import ABC, abstractmethod
import gymnasium as gym
import numpy as np

class DRLEnvironment(ABC):
    # Base class to handle agent interactions with the actual Environment
    # Description: This class is used by all of our algorithms to interact with the environment.
    #               Therefore, wrap your environment with this class to use it with this library.

    # Observes the environment and returns the state
    # Returns: state, reward, frame
    @abstractmethod
    def observe(self):
        pass

    # Calculates the reward from the state
    # Returns: New Reward
    @abstractmethod
    def calculate_reward(self, **kwargs):
        pass

    # Preprocesses a state (Default, returns state itself)
    # Returns: preprocessed state
    def preprocess_state(self, state):
        return state

    # Returns: Random action
    def get_random_action(self):
        pass

    # Takes an action
    @abstractmethod
    def take_action(self, action):
        pass

    # Checks whether the episode has finished or not
    # Returns: True or False based on whether the episode has finished or not.
    @abstractmethod
    def is_episode_finished(self):
        pass

    # Resets the environment for the next episode
    @abstractmethod
    def reset(self):
        pass

    # Closes the environment
    @abstractmethod
    def close(self):
        pass


class GymEnvironment(DRLEnvironment):
    # Implementation to interface with Gym Environments
    # Description: This is a default implementation provided to interface with OpenAI Gym Environments since
    #               OpenAI Gym is turning out to be the first choice to use and implement RL Environments

    def __init__(self, env):
        self.env = env
        self.terminated = False
        self.truncated = False
        self.reward = 0
        self.preprocessed_state = None
        self.state, _ = self.env.reset()

    # Returns the current state from observation, reward, frame
    def observe(self):
        frame = self.env.render()
        self.preprocessed_state = self.preprocess_state(self.state)
        self.reward = self.calculate_reward()
        return self.preprocessed_state, self.reward, frame

    # Takes an action
    def take_action(self, action):
        self.state, self.reward, self.terminated, self.truncated, _ = self.env.step(action)

    # Defaults to returning gym reward
    def calculate_reward(self, **kwargs):
        return self.reward

    def get_random_action(self):
        return self.env.action_space.sample()

    # Checks whether the episode has finished or not
    def is_episode_finished(self):
        return self.terminated or self.truncated

    def close(self):
        self.env.close()

    def reset(self):
        self.terminated = False
        self.truncated = False
        self.state, _ = self.env.reset()


# Creating Custom Gym Environment wrapper to change the random action function
class MountainCarContinuousEnvironment(GymEnvironment):
    def __init__(self, env: gym.Env, std):
        super(MountainCarContinuousEnvironment, self).__init__(env)
        self.theta = 0.15
        self.mean = np.zeros(1)
        self.std_dev = std * np.ones(1)
        self.dt = 1e-2
        self.x = np.zeros_like(self.mean)

    def get_random_action(self):
        # return random.normal(shape=(1,), mean=0.0, stddev=0.2, dtype=float32)
        self.x = (
                self.x
                + self.theta * (self.mean - self.x) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        return self.x

    def take_action(self, action):
        action = np.clip(action, -1, 1)
        super(MountainCarContinuousEnvironment, self).take_action(action)

    def calculate_reward(self):
        # reward = self.reward
        vel = abs(self.state[1]) * 1_000
        if vel < 0.1:
            vel = 0.1

        reward = (self.state[0] - 0.5) / vel
        if self.state[0] >= 0.5:
            reward = 100.0
        return reward
