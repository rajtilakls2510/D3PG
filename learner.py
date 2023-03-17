import multiprocessing
import os
import json
import threading
from signal import signal, SIGINT, SIGTERM
import rpyc
import tensorflow as tf
from rpyc.utils.helpers import classpartial
from rpyc.utils.server import ThreadedServer
from tensorflow.keras.models import clone_model, load_model


# ======================= Learner Main Process =========================================


class LearnerCoordinator:
    # Main Learner process is started with the object of this class to start the learner system
    def __init__(self, learner_parameters, config):
        self.reference_holders = {"algo_process": None, "accum_process": None, "param_process": None}
        self.connection_holders = {"algo": None, "accum": None, "param": None, "actor_coords": []}
        self.lcs_server = None
        self.config = config
        self.learner_parameters = learner_parameters
        self.config["actor_coords"] = []
        self.config["start_request_sent"] = False

    def process_terminator(self, signum, frame):
        print("Terminating Learner and Actor systems")

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
        self.lcs_server.close()
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
                                                                         args=(self.learner_parameters, self.config),
                                                                         name='Algorithm')
        self.reference_holders["algo_process"].start()

        # Starting Data Accumulator Process
        print("Starting Data Accumulator Process")
        self.reference_holders["accum_process"] = multiprocessing.Process(target=Pusher.process_starter,
                                                                          args=(self.config,),
                                                                          name='DataAccumulator')
        self.reference_holders["accum_process"].start()

        # Starting Parameter Server Process
        print("Starting Parameter Server Process")
        self.reference_holders["param_process"] = multiprocessing.Process(target=ParameterMain.process_starter,
                                                                          args=(self.config,),
                                                                          name='ParameterServer')
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

    def __init__(self, learner_parameters):
        self.step_counter = 1
        self.agent_path = learner_parameters["agent_path"]
        self.actor_network, self.critic_network = learner_parameters["network_creator"]()
        if self.actor_network is None:
            self.actor_target_network = None
        else:
            self.actor_target_network = clone_model(self.actor_network)

        if self.critic_network is None:
            self.critic_target_network = None
        else:
            self.critic_target_network = clone_model(self.critic_network)
        print(self.actor_network.get_weights())
        self.min_replay_transitions = learner_parameters["min_replay_transitions"]
        self.replay_buffer = learner_parameters["replay_buffer"](learner_parameters["replay_buffer_size"], continuous_actions=True)
        self.discount_factor = tf.convert_to_tensor(learner_parameters["discount_factor"])
        self.tau = tf.convert_to_tensor(learner_parameters["tau"])
        self.critic_loss = learner_parameters["critic_loss"]()

    @classmethod
    def start_as_server(cls, as_server):
        print("Starting Algorithm Server")
        as_server.start()

    @classmethod
    def process_starter(cls, learner_parameters, config):
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
        DDPGLearner.ddpg_learner_object = DDPGLearner(learner_parameters)

        # Waiting for start confirmation from LC
        start_event.wait()

        # After confirmation from LC, start training
        DDPGLearner.ddpg_learner_object.train(config)

    @classmethod
    def process_terminator(cls, signum, frame):
        DDPGLearner.ddpg_learner_object.close()
        DDPGLearner.lc_connection.close()
        DDPGLearner.as_server.close()
        exit(0)

    def push_parameters(self, parameter_server_conn):
        actor_weights = self.actor_network.get_weights()
        for i in range(len(actor_weights)):
            actor_weights[i] = tf.io.serialize_tensor(tf.constant(actor_weights[i])).numpy()
        critic_weights = self.critic_network.get_weights()
        for i in range(len(critic_weights)):
            critic_weights[i] = tf.io.serialize_tensor(tf.constant(critic_weights[i])).numpy()
        try:
            parameter_server_conn.root.set_params(actor_weights, critic_weights)
        except Exception as e:
            print(e)
            # Ignore if parameter server is not available (It will be available when LC restarts the process hopefully)

    def train(self, config):
        # Loading
        self.load(self.agent_path)
        # Initialize Parameter Server
        parameter_server_conn = rpyc.connect("localhost", port=config["param_server_port"])
        try:
            parameter_server_conn.root.set_nnet_arch(json.dumps(self.actor_network.get_config()), json.dumps(self.critic_network.get_config()))
        except Exception as e:
            print(e)
            # Ignore if parameter server is not available (It will be available when LC restarts the process hopefully)
        self.push_parameters(parameter_server_conn)

        # TODO: Training Logic

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
        try:
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
        except:
            # If actor or critic networks are not found, work with current actor and critic networks
            pass
        self.replay_buffer.load(os.path.join(path, "replay"))

    def close(self):
        # Release any resources here
        pass


# ======================= Parameter Server Process =========================================

@rpyc.service
class ParameterService(rpyc.Service):

    # Parameter Service serves the parameters for the learner
    def __init__(self, start_event, arch_push_event, first_param_push_event):
        self.start_event = start_event
        self.arch_push_event = arch_push_event
        self.first_param_push_event = first_param_push_event

    @rpyc.exposed
    def start_work(self):
        self.start_event.set()

    @rpyc.exposed
    def some2(self):
        self.arch_push_event.set()
        self.first_param_push_event.set()


    @rpyc.exposed
    def get_nnet_arch(self):
        self.arch_push_event.wait()
        return ParameterMain.actor_config, ParameterMain.critic_config

    @rpyc.exposed
    def get_params(self):
        self.first_param_push_event.wait()
        return ParameterMain.actor_params, ParameterMain.critic_params

    @rpyc.exposed
    def set_nnet_arch(self, actor_config, critic_config):
        ParameterMain.actor_config = actor_config
        ParameterMain.critic_config = critic_config
        self.arch_push_event.set()

    @rpyc.exposed
    def set_params(self, actor_params, critic_params):
        ParameterMain.actor_params = self.rpyc_deep_copy(actor_params)
        ParameterMain.critic_params = self.rpyc_deep_copy(critic_params)
        self.first_param_push_event.set()

    def rpyc_deep_copy(self, obj):
        """
        Makes a deep copy of netref objects that come as a result of RPyC remote method calls.
        When RPyC client obtains a result from the remote method call, this result may contain
        non-scalar types (List, Dict, ...) which are given as a wrapper class (a netref object).
        This class does not have all the standard attributes (e.g. dict.tems() does not work)
        and in addition the objects only exist while the connection is active (are weekly referenced).
        To have a retuned value represented by python's native datatypes and to by able to use it
        after the connection is terminated, this routine makes a recursive copy of the given object.
        Currently, only `list` and `dist` types are supported for deep_copy, but other types may be
        added easily.
        Note there is allow_attribute_public option for RPyC connection, which may solve the problem too,
        but it have not worked for me.
        Example:
            s = rpyc.connect(host1, port)
            result = rpyc_deep_copy(s.root.remote_method())
            # if result is a Dict:
            for k,v in result.items(): print(k,v)
        """
        if (isinstance(obj, list)):
            copied_list = []
            for value in obj: copied_list.append(self.rpyc_deep_copy(value))
            return copied_list
        elif (isinstance(obj, dict)):
            copied_dict = {}
            for key in obj: copied_dict[key] = self.rpyc_deep_copy(obj[key])
            return copied_dict
        else:
            return obj

class ParameterMain:
    ps_server = None
    lc_connection = None
    parameter_main_object = None

    actor_config = None
    critic_config = None
    actor_params = None
    critic_params = None

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
        arch_push_event = threading.Event()
        first_param_push_event = threading.Event()
        ps_service = classpartial(ParameterService, start_event, arch_push_event, first_param_push_event)
        ParameterMain.ps_server = ThreadedServer(ps_service, port=config["param_server_port"], protocol_config={'allow_public_attrs': True,})
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
    def process_terminator(cls, signum, frame):
        ParameterMain.parameter_main_object.close()
        ParameterMain.lc_connection.close()
        ParameterMain.ps_server.close()
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
    def process_terminator(cls, signum, frame):
        Pusher.pusher_object.close()
        Pusher.lc_connection.close()
        Pusher.das_server.close()
        exit(0)

    def start(self):
        # Start thread-safe queue and that fun stuff
        pass

    def close(self):
        # Release any resources here
        pass
