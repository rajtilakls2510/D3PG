import rpyc
import tensorflow as tf


class DDPGActor:
    def __init__(self):
        self.env = None
        self.actor_network = None
        self.critic_network = None

    def set_env(self, env):
        self.env = env

    def get_action(self, state, explore=0.0):
        state = tf.expand_dims(state, axis=0)
        action = self.actor_network(state)
        explored = tf.constant(False)
        if tf.random.uniform(shape=(), maxval=1) < explore:
            action = action + tf.convert_to_tensor(self.env.get_random_action(), tf.float32)
            explored = tf.constant(True)
        value = self.critic_network([state, action])
        return action[0], value[0][0], explored

    def get_values(self, states):
        return self.critic_network([states, self.actor_network(states)])


class ActorCoordinator:
    # Main Actor process is started with the object of this class to start the actor system
    pass


class ActorCoordinatorService(rpyc.Service):
    # The main actor service
    pass

