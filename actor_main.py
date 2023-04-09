import actor
import environment
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
class LunarLanderContinuousEnvironment(environment.GymEnvironment):

    def __init__(self, env: gym.Env):
        super(LunarLanderContinuousEnvironment, self).__init__(env)
        self.theta = 0.15
        self.mean = np.zeros(2)
        self.std_dev = float(0.5) * np.ones(2)
        self.dt = 1e-2
        self.x = np.zeros_like(self.mean)

    def get_randomized_action(self):
        self.x = (
                self.x
                + self.theta * (self.mean - self.x) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        return self.x

    def take_action(self, action):
        action = np.clip(action, -1, 1)
        super(LunarLanderContinuousEnvironment, self).take_action(action)


def env_creator():
    return LunarLanderContinuousEnvironment(gym.make("LunarLander-v2", continuous=True, render_mode="rgb_array"))


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
    "exploration": 0.5,
    "n_fetch": 1,
    "n_push": 30,
    "max_executors": 10,
    "show_acting": False,
}

if __name__ == "__main__":
    actor_coord = actor.ActorCoordinator(env_creator, config, actor_parameters)
    actor_coord.start()
    print("Actor System Started")
    # You gotta keep working for signals to be received
    while True:
        pass
