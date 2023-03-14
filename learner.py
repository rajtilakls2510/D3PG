import multiprocessing
import os
import tensorflow as tf
import threading
from rpyc.utils.server import ThreadedServer
from tensorflow.keras.models import clone_model, load_model
import replaybuffer
import rpyc
from rpyc.utils.helpers import classpartial
from signal import signal, SIGINT, SIGTERM


# ======================= Learner Main Process =========================================


class LearnerCoordinator:
    # Main Learner process is started with the object of this class to start the learner system
    def __init__(self, network_creators, config):
        self.reference_holders = {"algo_process": None, "accum_process": None, "param_process": None}
        self.connection_holders = {"algo": None, "accum": None, "param": None, "actor_coords": []}
        self.lcs_server = None
        self.config = config
        self.network_creators = network_creators
        self.config["actor_coords"] = []
        self.config["start_request_sent"] = False

    def process_terminator(self, signum, frame):
        print("Terminating Learner and Actor systems")
        self.lcs_server.close()
        for actor_conn in self.connection_holders["actor_coords"]:
            try:
                actor_conn.root.stop()
            except:
                pass
        self.connection_holders["algo"].close()
        self.connection_holders["accum"].close()
        self.connection_holders["param"].close()
        self.reference_holders["algo_process"].terminate()
        self.reference_holders["accum_process"].terminate()
        self.reference_holders["param_process"].terminate()
        self.reference_holders["algo_process"].join()
        self.reference_holders["accum_process"].join()
        self.reference_holders["param_process"].join()
        exit(0)

    def start_lcs_server(self):
        print("Starting Learner Coordinator Server...")
        self.lcs_server.start()

    def start(self):
        events = {"algo": threading.Event(), "accum": threading.Event(), "param": threading.Event(),
                  "actors": threading.Event()}

        signal(SIGINT, self.process_terminator)
        signal(SIGTERM, self.process_terminator)

        # Starting LCS Server on a different thread
        lcs = classpartial(LearnerCoordinatorService, events, self.reference_holders, self.config,
                           self.connection_holders)
        self.lcs_server = ThreadedServer(lcs, port=self.config["lcs_server_port"])
        t1 = threading.Thread(target=self.start_lcs_server)
        t1.start()

        # Starting Algorithm Process
        print("Starting Algorithm Process")
        self.reference_holders["algo_process"] = multiprocessing.Process(target=DDPGLearner.process_starter,
                                                                         args=(self.network_creators, self.config))
        self.reference_holders["algo_process"].start()

        # Starting Data Accumulator Process
        print("Starting Data Accumulator Process")
        self.reference_holders["accum_process"] = multiprocessing.Process(target=Pusher.process_starter,
                                                                          args=(self.config,))
        self.reference_holders["accum_process"].start()

        # Starting Parameter Server Process
        print("Starting Parameter Server Process")
        self.reference_holders["param_process"] = multiprocessing.Process(target=ParameterMain.process_starter,
                                                                          args=(self.config,))
        self.reference_holders["param_process"].start()

        # Await confirmation from 4 workers
        events["algo"].wait()
        events["accum"].wait()
        events["param"].wait()
        # print("Waiting for one Actor Coordinator to join")
        # events["actors"].wait()

        # Connecting to all workers
        self.connection_holders["algo"] = rpyc.connect("localhost", port=self.config["algo_server_port"])
        self.connection_holders["accum"] = rpyc.connect("localhost", port=self.config["accum_server_port"])
        self.connection_holders["param"] = rpyc.connect("localhost", port=self.config["param_server_port"])

        # Sending start request to everyone
        self.connection_holders["algo"].root.start_work()
        self.connection_holders["accum"].root.start_work()
        self.connection_holders["param"].root.start_work()
        for conn in self.connection_holders["actor_coords"]:
            conn.root.start_work()
        self.config["start_request_sent"] = True

    def monitor_system(self):
        # Monitors the system. Makes sure all child processes are up and running. It is called after start_system is
        # called
        pass


@rpyc.service
class LearnerCoordinatorService(rpyc.Service):
    # The main learner service

    def __init__(self, events, reference_holders, config, connection_holders):
        self.events = events
        self.reference_holders = reference_holders
        self.config = config
        self.connection_holders = connection_holders

    @rpyc.exposed
    def component_started_confirmation(self, signature, info=None):
        if signature == "actors":
            self.config["actor_coords"].append(info)
            conn = rpyc.connect(info['host'], port=info['port'])
            self.connection_holders["actor_coords"].append(conn)
            if self.config["start_request_sent"]:
                conn.root.start_work()
            print(f"Actor connected with host: {info['host']} port:{info['port']}")
        else:
            self.events[signature].set()


# ======================= Algorithm Process =========================================

@rpyc.service
class AlgorithmService(rpyc.Service):
    # Serves the requests about the algorithm
    def __init__(self, start_event):
        self.start_event = start_event

    @rpyc.exposed
    def start_work(self):
        self.start_event.set()


class DDPGLearner:
    as_server = None
    lc_connection = None
    ddpg_learner_object = None

    def __init__(self, network_creators, learn_after_steps=1,
                 replay_size=1000, exploration=0.1, min_exploration=0.0, exploration_decay=1.1,
                 exploration_decay_after=100, discount_factor=0.9, tau=0.001):
        super().__init__()
        self.actor_network, self.critic_network = network_creators()
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

    @classmethod
    def start_as_server(cls, as_server):
        print("Starting Algorithm Server")
        as_server.start()

    @classmethod
    def process_starter(cls, network_creators, config):
        print(f"Algorithm Process Started: {os.getpid()}")

        signal(SIGINT, DDPGLearner.process_terminator)
        signal(SIGTERM, DDPGLearner.process_terminator)

        # Starting Algorithm Server
        start_event = threading.Event()
        as_service = classpartial(AlgorithmService, start_event)
        DDPGLearner.as_server = ThreadedServer(as_service, port=config["algo_server_port"])
        t1 = threading.Thread(target=DDPGLearner.start_as_server, args=[DDPGLearner.as_server])
        t1.start()

        # Sending confirmation to LC about successful process start
        DDPGLearner.lc_connection = rpyc.connect("localhost", port=config["lcs_server_port"])
        DDPGLearner.lc_connection.root.component_started_confirmation("algo")
        DDPGLearner.ddpg_learner_object = DDPGLearner(network_creators)

        # Waiting for start confirmation from LC
        start_event.wait()

        # After confirmation from LC, start training
        DDPGLearner.ddpg_learner_object.train()

    @classmethod
    def process_terminator(cls):
        DDPGLearner.ddpg_learner_object.close()
        DDPGLearner.as_server.close()
        DDPGLearner.lc_connection.close()
        exit(0)

    def train(self):
        # TODO: Write training logic
        pass

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
        self.actor_network.save(os.path.join(path, "actor_network"))
        self.actor_target_network.save(os.path.join(path, "actor_target_network"))
        self.critic_network.save(os.path.join(path, "critic_network"))
        self.critic_target_network.save(os.path.join(path, "critic_target_network"))
        self.replay_buffer.save(os.path.join(path, "replay"))

    def load(self, path=""):
        self.actor_network = load_model(os.path.join(path, "actor_network"))
        self.critic_network = load_model(os.path.join(path, "critic_network"))
        try:
            self.actor_target_network = load_model(os.path.join(path, "actor_target_network"))
        except:
            if self.actor_network is not None:
                self.actor_target_network = clone_model(self.actor_network)
        try:
            self.critic_target_network = load_model(os.path.join(path, "critic_target_network"))
        except:
            if self.critic_network is not None:
                self.critic_target_network = clone_model(self.critic_network)
        self.replay_buffer.load(os.path.join(path, "replay"))

    def close(self):
        # Release any resources here
        pass


# ======================= Parameter Server Process =========================================

@rpyc.service
class ParameterService(rpyc.Service):
    # Parameter Service serves the parameters for the learner
    def __init__(self, start_event):
        self.start_event = start_event

    @rpyc.exposed
    def start_work(self):
        self.start_event.set()


class ParameterMain:
    ps_server = None
    lc_connection = None
    parameter_main_object = None

    @classmethod
    def start_ps_server(cls, ps_server):
        print("Starting Parameter Server")
        ps_server.start()

    @classmethod
    def process_starter(cls, config):
        print(f"Parameter Server Process Started: {os.getpid()}")

        signal(SIGINT, ParameterMain.process_terminator)
        signal(SIGTERM, ParameterMain.process_terminator)

        # Starting Parameter Server
        start_event = threading.Event()
        ps_service = classpartial(ParameterService, start_event)
        ParameterMain.ps_server = ThreadedServer(ps_service, port=config["param_server_port"])
        t1 = threading.Thread(target=ParameterMain.start_ps_server, args=[ParameterMain.ps_server])
        t1.start()

        # Sending confirmation to LC about successful process start
        ParameterMain.lc_connection = rpyc.connect("localhost", port=config["lcs_server_port"])
        ParameterMain.lc_connection.root.component_started_confirmation("param")
        ParameterMain.parameter_main_object = ParameterMain()
        # Not sure what to do with the object but still keeping it for potential future use

        # Waiting for start confirmation from LC
        start_event.wait()
        t1.join()

    @classmethod
    def process_terminator(cls):
        ParameterMain.parameter_main_object.close()
        ParameterMain.ps_server.close()
        ParameterMain.lc_connection.close()
        exit(0)

    def close(self):
        # Release any resources here
        pass


# ======================= Data Accumulator Process =========================================

@rpyc.service
class DataAccumulatorService(rpyc.Service):
    # Accumulates different kinds of data into a thread-safe queue
    def __init__(self, start_event):
        self.start_event = start_event

    @rpyc.exposed
    def start_work(self):
        self.start_event.set()


class Pusher:
    # Pushes the data in the queue into algorithm process
    das_server = None
    lc_connection = None
    pusher_object = None

    @classmethod
    def start_das_server(cls, das_server):
        print("Starting Data Accumulator Server")
        das_server.start()

    @classmethod
    def process_starter(cls, config):
        print(f"Data Accumulator Process Started: {os.getpid()}")

        signal(SIGINT, Pusher.process_terminator)
        signal(SIGTERM, Pusher.process_terminator)

        # Starting Data Accumulator Server
        start_event = threading.Event()
        das_service = classpartial(DataAccumulatorService, start_event)
        Pusher.das_server = ThreadedServer(das_service, port=config["accum_server_port"])
        t1 = threading.Thread(target=Pusher.start_das_server, args=[Pusher.das_server])
        t1.start()

        # Sending confirmation to LC about successful process start
        Pusher.lc_connection = rpyc.connect("localhost", port=config["lcs_server_port"])
        Pusher.lc_connection.root.component_started_confirmation("accum")
        Pusher.pusher_object = Pusher()

        # Waiting for start confirmation from LC
        start_event.wait()

        # After confirmation from LC, start pusher
        Pusher.pusher_object.start()  # TODO: Add pusher logic

    @classmethod
    def process_terminator(cls):
        Pusher.pusher_object.close()
        Pusher.das_server.close()
        Pusher.lc_connection.close()
        exit(0)

    def start(self):
        # Start thread-safe queue and that fun stuff
        pass

    def close(self):
        # Release any resources here
        pass
