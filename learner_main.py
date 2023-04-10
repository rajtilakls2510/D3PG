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
    state_input = Input(shape=(2,))
    x = Dense(64, activation="relu")(state_input)
    x = Dense(64, activation="relu")(x)
    output = Dense(1, activation='tanh', kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003))(x)
    actor_network = Model(inputs=state_input, outputs=output)
    actor_network.compile(optimizer=Adam(learning_rate=0.0005))

    # Critic Network
    state_input = Input(shape=(2,))
    action_input = Input(shape=(1,))
    x1 = Dense(32, activation='relu')(state_input)
    x2 = Dense(32, activation='relu')(action_input)
    x = Concatenate()([x1, x2])
    x = Dense(64, activation="relu")(x)
    output = Dense(1, activation="linear", kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003))(x)
    critic_network = Model(inputs=[state_input, action_input], outputs=output)
    critic_network.compile(optimizer=Adam(learning_rate=0.001))

    return actor_network, critic_network


learner_parameters = {
    "agent_path": "mountain_car_cont_agent2",
    "network_creator": network_creator,
    "n_learn": 1000,
    "n_persis": 1000,
    "batch_size": 64,
    # "min_replay_transitions": 1000,
    "replay_buffer": replaybuffer.UniformReplay,
    "replay_buffer_size": 100000,
    "discount_factor": 0.99,
    "tau": 0.001,
    "critic_loss": tf.keras.losses.MeanSquaredError,
    "actor_learn": 4
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