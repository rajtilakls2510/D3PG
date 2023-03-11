import tensorflow as tf, os
from tensorflow.keras.models import clone_model, load_model
import replaybuffer, rpyc


class DDPGLearner:

    def __init__(self, actor_network: tf.keras.Model = None, critic_network: tf.keras.Model = None, learn_after_steps=1,
                 replay_size=1000, exploration=0.1, min_exploration=0.0, exploration_decay=1.1,
                 exploration_decay_after=100, discount_factor=0.9, tau=0.001):
        super().__init__()
        self.actor_network = actor_network
        self.critic_network = critic_network
        if self.actor_network is None:
            self.actor_target_network = None
        else:
            self.actor_target_network = clone_model(self.actor_network)

        if self.critic_network is None:
            self.critic_target_network = None
        else:
            self.critic_target_network = clone_model(self.critic_network)

        self.learn_after_steps = learn_after_steps
        self.replay_buffer = replaybuffer.UniformReplay(replay_size, continuous=True)
        self.discount_factor = tf.convert_to_tensor(discount_factor)
        self.exploration = exploration
        self.min_exploration = min_exploration
        self.exploration_decay = exploration_decay
        self.exploration_decay_after = exploration_decay_after
        self.tau = tf.convert_to_tensor(tau)
        self.step_counter = 1
        self.critic_loss = tf.keras.losses.MeanSquaredError()

    @tf.function
    def _train_step(self, current_states, actions, rewards, next_states):

        targets = tf.expand_dims(rewards, axis=1) + self.discount_factor * self.critic_target_network(
            [next_states, self.actor_target_network(next_states)])

        with tf.GradientTape() as critic_tape:
            critic_value = self.critic_network([current_states, actions])
            critic_loss = self.critic_loss(targets, critic_value)

        critic_grads = critic_tape.gradient(critic_loss, self.critic_network.trainable_weights)
        self.critic_network.optimizer.apply_gradients(zip(critic_grads, self.critic_network.trainable_weights))

        with tf.GradientTape() as actor_tape:
            actor_loss = -tf.reduce_mean(self.critic_network([current_states, self.actor_network(current_states)]))

        actor_grads = actor_tape.gradient(actor_loss, self.actor_network.trainable_weights)
        self.actor_network.optimizer.apply_gradients(zip(actor_grads, self.actor_network.trainable_weights))

    @tf.function
    def update_targets(self, target_weights, weights, tau):
        for (target_w, w) in zip(target_weights, weights):
            target_w.assign(tau * w + (1 - tau) * target_w)


    def save(self, path=""):
        self.q_network.save(os.path.join(path, "q_network"))
        self.target_network.save(os.path.join(path, "target_network"))
        self.replay_buffer.save(os.path.join(path, "replay"))

    def load(self, path=""):
        self.q_network = load_model(os.path.join(path, "q_network"))
        try:
            self.target_network = load_model(os.path.join(path, "target_network"))
        except:
            if self.q_network is not None:
                self.target_network = clone_model(self.q_network)
        self.replay_buffer.load(os.path.join(path, "replay"))


class LearnerCoordinator:
    # Main Learner process is started with the object of this class to start the learner system
    pass


class LearnerCoordinatorService(rpyc.Service):
    # The main learner service
    pass


class ParameterService(rpyc.Service):
    # Parameter Service serves the parameters for the learner
    pass


class DataAccumulatorService(rpyc.Service):
    # Accumulates different kinds of data into a thread-safe queue
    pass


class Pusher:
    # Pushes the data in the queue into algorithm process
    pass

