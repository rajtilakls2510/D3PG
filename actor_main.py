import actor
import environment
import logger
import os
# Set this to not use Tensorflow-GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# import tensorflow as tf
import numpy as np
import gymnasium as gym
import random
import tensorflow as tf


# Set memory_growth option to True otherwise tensorflow will eat up all GPU memory
try:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


# Creating Custom Gym Environment wrapper to change the random action function
class MountainCarContinuousEnvironment(environment.GymEnvironment):
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


def env_creator(std):
    return MountainCarContinuousEnvironment(gym.make("MountainCarContinuous-v0", render_mode = "rgb_array"), std)


config = {
    "num_actors": 5,
    "lcs_server_host": "localhost",
    "lcs_server_port": 18861,
    "acs_server_host": "localhost",
    "acs_server_port": 18865,
    "param_server_host": "localhost",
    "param_server_port": 18864,
    "accum_server_host": "localhost",
    "accum_server_port": 18863
}

actor_parameters = {
    "exploration": 1.0,
    "n_fetch": 20,
    "n_push": 30,
    "max_executors": 10,
    "show_acting": False,
    "logs": [logger.EpisodeLengthLogger, logger.TotalRewardLogger, logger.EpisodeTimeLogger],
    "std": [0.2, 0.3, 0.4]
}

if __name__ == "__main__":
    actor_coord = actor.ActorCoordinator(env_creator, config, actor_parameters)
    actor_coord.start()
    print("Actor System Started")
    # You gotta keep working for signals to be received
    while True:
        pass
