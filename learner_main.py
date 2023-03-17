import learner
import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras.layers as layers
import replaybuffer

# Set memory_growth option to True otherwise tensorflow will eat up all GPU memory
try:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

def network_creator():
    # Testing with actor and critic
    inp = layers.Input(shape=(512, ))
    x = layers.Dense(64, activation = "relu")(inp)
    x = layers.Dense(64, activation = "relu")(x)
    x = layers.Dense(64, activation = "relu")(x)
    out = layers.Dense(10, activation = "softmax")(x)

    actor = Model(inputs=inp, outputs=out)

    actor.compile(loss = "categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    critic = actor
    return actor, critic


learner_parameters = {
    "agent_path": "some_agent",
    "network_creator": network_creator,
    "min_replay_transitions": 1000,
    "replay_buffer": replaybuffer.UniformReplay,
    "replay_buffer_size": 100000,
    "discount_factor": 0.99,
    "tau": 0.001,
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