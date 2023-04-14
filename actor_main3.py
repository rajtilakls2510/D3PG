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


def env_creator(std):
    return environment.MountainCarContinuousEnvironment(gym.make("MountainCarContinuous-v0", render_mode = "rgb_array"), std)


config = {
    "num_actors": 1,
    "lcs_server_host": "localhost",
    "lcs_server_port": 18861,
    "acs_server_host": "localhost",
    "acs_server_port": 18867,
    "param_server_host": "localhost",
    "param_server_port": 18864,
    "accum_server_host": "localhost",
    "accum_server_port": 18863
}

actor_parameters = {
    "mode": "eval",
    "exploration": 0.0,
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
