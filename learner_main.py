import learner
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Add, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform
import replaybuffer
import numpy as np

# Set memory_growth option to True otherwise tensorflow will eat up all GPU memory
try:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

tf.random.set_seed(tf.random.uniform(shape=(1,), minval=0, maxval=1000, dtype=tf.int32))
np.random.seed(np.random.randint(0, 1000))


def network_creator():
    # Actor Network
    state_input = Input(shape=(27,))
    x = Dense(400, activation="relu")(state_input)
    x = BatchNormalization()(x)
    x = Dense(300, activation="relu")(x)
    x = BatchNormalization()(x)
    # x = Dense(256)(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=0.0)(x)
    output = Dense(8, activation="tanh", kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003))(x)
    actor_network = Model(inputs=state_input, outputs=output)
    actor_network.compile(optimizer=Adam(learning_rate=0.001))

    # Critic Network
    state_input = Input(shape=(27,))
    x1 = Dense(400, activation="relu")(state_input)
    x1 = BatchNormalization()(x1)

    action_input = Input(shape=(8,))
    x2 = Dense(400, activation="relu")(action_input)

    x = Add()([x1, x2])
    x = Dense(300, activation="relu")(x)
    output = Dense(1, activation="linear", kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003))(x)
    critic_network1 = Model(inputs=[state_input, action_input], outputs=output)
    critic_network1.compile(optimizer=Adam(learning_rate=0.001))

    critic_network2 = Model.from_config(critic_network1.get_config())
    critic_network2.compile(optimizer=Adam(learning_rate=0.001))

    return actor_network, critic_network1, critic_network2

# 2345
# 3707
# 1262
# 233
# 2604
learner_parameters = {
    "agent_path": "ant_agent",
    "network_creator": network_creator,
    "n_learn": 1_000,
    "n_persis": 1200,
    "batch_size": 100,
    "min_replay_size": 1000,
    "replay_buffer": replaybuffer.UniformReplay,
    "replay_buffer_size": 300_000,
    "discount_factor": 0.99,
    "tau": 0.005,
    "critic_loss": tf.keras.losses.MeanSquaredError,
    "actor_learn": 2,
    "target_noise_std": 0.2,
    "target_noise_clipvalue": 0.5
}


config = {
    "lcs_server_port": 18861,
    "algo_server_port": 18862,
    "accum_server_port": 18863,
    "param_server_port": 18864
}
# print(network_creator()[0].get_config())
if __name__ == "__main__":
    learner_coord = learner.LearnerCoordinator.get_instance(learner_parameters, config)
    learner_coord.start()
    print("Learner System Started")

    learner_coord.monitor_system()

