import learner
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform
import replaybuffer

# Set memory_growth option to True otherwise tensorflow will eat up all GPU memory
try:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

def network_creator():
    # Actor Network
    state_input = Input(shape=(8,))
    x = Dense(256, activation="sigmoid")(state_input)
    x = BatchNormalization()(x)
    x = Dense(256, activation="sigmoid")(x)
    x = BatchNormalization()(x)
    output = Dense(2, activation="tanh", kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003))(x)
    actor_network = Model(inputs=state_input, outputs=output)
    actor_network.compile(optimizer=Adam(learning_rate=0.001))

    # Critic Network
    state_input = Input(shape=(8,))
    x1 = Dense(192, activation="sigmoid")(state_input)
    x1 = BatchNormalization()(x1)

    action_input = Input(shape=(2,))
    x2 = Dense(64, activation="sigmoid")(action_input)

    x = Concatenate()([x1, x2])
    x = Dense(256, activation="sigmoid")(x)
    output = Dense(1, activation="linear", kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003))(x)
    critic_network = Model(inputs=[state_input, action_input], outputs=output)
    critic_network.compile(optimizer=Adam(learning_rate=0.001))
    return actor_network, critic_network


learner_parameters = {
    "agent_path": "some_agent",
    "network_creator": network_creator,
    "n_learn": 100,
    "n_persis": 10000,
    "batch_size": 64,
    # "min_replay_transitions": 1000,
    "replay_buffer": replaybuffer.UniformReplay,
    "replay_buffer_size": 100000,
    "discount_factor": 0.99,
    "tau": 0.0001,
    "critic_loss": tf.keras.losses.MeanSquaredError
}


config = {
    "lcs_server_port": 18861,
    "algo_server_port": 18862,
    "accum_server_port": 18863,
    "param_server_port": 18864
}

if __name__ == "__main__":
    learner_coord = learner.LearnerCoordinator(learner_parameters, config)
    learner_coord.start()
    print("Learner System Started")

    # You gotta keep working for signals to be received
    while True:
        pass