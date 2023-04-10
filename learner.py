import multiprocessing
import os
import json
import base64
import threading
import time
from signal import signal, SIGINT, SIGTERM

import pandas as pd
import rpyc
import tensorflow as tf
from queue import Queue
from rpyc.utils.helpers import classpartial
from rpyc.utils.server import ThreadedServer
from tensorflow.keras.models import clone_model, load_model
from tensorflow.keras import Model



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
                                                                          args=(self.learner_parameters, self.config),
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
    def __init__(self, start_event, ddpg_learner):
        self.start_event = start_event
        self.ddpg_learner = ddpg_learner

    @rpyc.exposed
    def start_work(self):
        self.start_event.set()

    def parse_and_insert(self, data):
        data_list = json.loads(data)
        t1, t2, total_transitions = 0, 0, 0
        from time import perf_counter
        for batch in data_list:
            s1 = perf_counter()
            batch = json.loads(batch)
            batch["current_state"] = tf.io.parse_tensor(base64.b64decode(batch["current_state"]), out_type=tf.float32)
            batch["action"] = tf.io.parse_tensor(base64.b64decode(batch["action"]), out_type=tf.float32)
            batch["reward"] = tf.io.parse_tensor(base64.b64decode(batch["reward"]), out_type=tf.float32)
            batch["next_state"] = tf.io.parse_tensor(base64.b64decode(batch["next_state"]), out_type=tf.float32)
            batch["terminated"] = tf.io.parse_tensor(base64.b64decode(batch["terminated"]), out_type=tf.bool)
            s2 = perf_counter()
            self.ddpg_learner.replay_buffer.insert_batch_transitions(batch)
            s3 = perf_counter()
            t1 += (s2 - s1)
            t2 += (s3 - s2)
            total_transitions += batch["current_state"].shape[0]
        print(f"Parsing: {t1} Inserting: {t2} Transitions: {total_transitions}")
        # print(f"Replay Size: {self.ddpg_learner.replay_buffer.current_states.size()}")

    @rpyc.exposed
    def push_replay_data(self, data):
        threading.Thread(target=self.parse_and_insert, args=(data,)).start()


class DDPGLearner:
    learner_object = None

    def __init__(self, learner_parameters):
        self.parameter_server_conn = None
        self.accum_server_conn = None
        self.as_server = None
        self.lc_connection = None
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
        self.n_learn = learner_parameters["n_learn"]
        self.n_persis = learner_parameters["n_persis"]
        self.batch_size = learner_parameters["batch_size"]
        # self.min_replay_transitions = learner_parameters["min_replay_transitions"]
        self.replay_buffer = learner_parameters["replay_buffer"](learner_parameters["replay_buffer_size"], continuous_actions=True)
        self.discount_factor = tf.convert_to_tensor(learner_parameters["discount_factor"])
        self.tau = tf.convert_to_tensor(learner_parameters["tau"])
        self.critic_loss = learner_parameters["critic_loss"]()
        self.actor_learn = learner_parameters["actor_learn"]

    def start_as_server(self):
        print("Starting Algorithm Server")
        self.as_server.start()

    @classmethod
    def process_starter(cls, learner_parameters, config):
        print(f"Algorithm Process Started: {os.getpid()}")

        signal(SIGINT, DDPGLearner.process_terminator)
        signal(SIGTERM, DDPGLearner.process_terminator)

        # Starting Algorithm Server
        start_event = threading.Event()
        learner = DDPGLearner(learner_parameters)
        DDPGLearner.learner_object = learner
        as_service = classpartial(AlgorithmService, start_event, learner)
        learner.as_server = ThreadedServer(as_service, port=config["algo_server_port"])
        t1 = threading.Thread(target=learner.start_as_server)
        t1.start()

        # Sending confirmation to LC about successful process start
        learner.lc_connection = rpyc.connect("localhost", port=config["lcs_server_port"])
        learner.lc_connection.root.component_started_confirmation("algo")

        # Waiting for start confirmation from LC
        start_event.wait()

        # After confirmation from LC, start training
        learner.train(config)

    @classmethod
    def process_terminator(cls, signum, frame):
        learner = DDPGLearner.learner_object
        learner.close()
        learner.lc_connection.close()
        learner.as_server.close()
        exit(0)

    def push_parameters(self):
        actor_weights = self.actor_network.get_weights()

        # Protocol for Sending:
        # - Serialize Tensor
        # - Base64 Encode it
        # - Convert list of parameters to json

        for i in range(len(actor_weights)):
            actor_weights[i] = base64.b64encode(tf.io.serialize_tensor(tf.convert_to_tensor(actor_weights[i])).numpy()).decode('ascii')
        critic_weights = self.critic_network.get_weights()
        for i in range(len(critic_weights)):
            critic_weights[i] = base64.b64encode(tf.io.serialize_tensor(tf.convert_to_tensor(critic_weights[i])).numpy()).decode('ascii')
        try:
            self.parameter_server_conn.root.set_params(json.dumps(actor_weights), json.dumps(critic_weights))
        except Exception as e:
            print(e)
            # Ignore if parameter server is not available (It will be available when LC restarts the process hopefully)

    def save_keras_model(self, model, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            f.write(json.dumps(model.get_config()))

        model.save_weights(os.path.join(path, "weights"))

    def load_keras_model(self,path):
        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.loads(f.read())
        model = Model().from_config(config)
        model.load_weights(os.path.join(path, "weights"))
        return model

    def train(self, config):
        # Loading
        self.load(self.agent_path)
        # Initialize Parameter Server
        self.parameter_server_conn = rpyc.connect("localhost", port=config["param_server_port"])
        self.accum_server_conn = rpyc.connect("localhost", port=config["accum_server_port"])
        self.parameter_server_conn.root.set_nnet_arch(json.dumps(self.actor_network.get_config()), json.dumps(self.critic_network.get_config()))
        self.push_parameters()

        persis_steps = 0
        while persis_steps <= self.n_persis:
            learn_steps = 0
            start = time.perf_counter()
            # Learning continuously without interruption for n_learn steps
            while learn_steps <= self.n_learn:
                current_states, actions, rewards, next_states, terminals = self.replay_buffer.sample_batch_transitions(
                    batch_size=self.batch_size)
                if current_states.shape[0] >= self.batch_size:
                    self._critic_train_step(current_states, actions, rewards, next_states)
                    if learn_steps % self.actor_learn == 0:
                        self._actor_train_step(current_states)
                        self.update_targets(self.actor_target_network.trainable_weights,
                                            self.actor_network.trainable_weights, self.tau)
                        self.update_targets(self.critic_target_network.trainable_weights,
                                            self.critic_network.trainable_weights, self.tau)
                learn_steps += 1
            end = time.perf_counter()
            self.push_parameters()
            self.accum_server_conn.root.collect_accum_data(persis_steps * self.n_learn)
            self.save(self.agent_path)
            persis_steps += 1
            print(f"persis: {persis_steps} time: {end-start}s")
            print(f"Persis: {persis_steps}")

    @tf.function
    def _critic_train_step(self, current_states, actions, rewards, next_states):
        targets = tf.expand_dims(rewards, axis=1) + self.discount_factor * self.critic_target_network(
            [next_states, self.actor_target_network(next_states)])

        with tf.GradientTape() as critic_tape:
            critic_value = self.critic_network([current_states, actions])
            critic_loss = self.critic_loss(targets, critic_value)

        critic_grads = critic_tape.gradient(critic_loss, self.critic_network.trainable_weights)
        self.critic_network.optimizer.apply_gradients(zip(critic_grads, self.critic_network.trainable_weights))

    @tf.function
    def _actor_train_step(self, current_states):

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
        # self.save_keras_model(self.actor_network, os.path.join(path, "actor_network"))
        # self.save_keras_model(self.actor_target_network, os.path.join(path, "actor_target_network"))
        # self.save_keras_model(self.critic_network, os.path.join(path, "critic_network"))
        # self.save_keras_model(self.critic_target_network, os.path.join(path, "critic_target_network"))

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
        #     self.actor_network = self.load_keras_model(os.path.join(path, "actor_network"))
        #     self.critic_network = self.load_keras_model(os.path.join(path, "critic_network"))
        #
        #     try:
        #         self.actor_target_network = self.load_keras_model(os.path.join(path, "actor_target_network"))
        #     except:
        #         if self.actor_network is not None:
        #             self.actor_target_network = clone_model(self.actor_network)
        #     try:
        #         self.critic_target_network = self.load_keras_model(os.path.join(path, "critic_target_network"))
        #     except:
        #         if self.critic_network is not None:
        #             self.critic_target_network = clone_model(self.critic_network)
        except Exception as e:
            # If actor or critic networks are not found, work with current actor and critic networks
            print(e)
        self.replay_buffer.load(os.path.join(path, "replay"))

    def close(self):
        if self.parameter_server_conn:
            self.parameter_server_conn.close()


# ======================= Parameter Server Process =========================================

@rpyc.service
class ParameterService(rpyc.Service):

    # Parameter Service serves the parameters for the learner
    def __init__(self, start_event, arch_push_event, first_param_push_event, param_object):
        self.start_event = start_event
        self.arch_push_event = arch_push_event
        self.first_param_push_event = first_param_push_event
        self.param_object = param_object

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
        return self.param_object.actor_config, self.param_object.critic_config

    @rpyc.exposed
    def get_params(self):
        self.first_param_push_event.wait()
        return self.param_object.actor_params, self.param_object.critic_params

    @rpyc.exposed
    def set_nnet_arch(self, actor_config, critic_config):
        self.param_object.actor_config = actor_config
        self.param_object.critic_config = critic_config
        self.arch_push_event.set()

    @rpyc.exposed
    def set_params(self, actor_params, critic_params):
        self.param_object.actor_params = actor_params
        self.param_object.critic_params = critic_params
        self.first_param_push_event.set()


class ParameterMain:
    parameter_main_object = None

    def __init__(self):
        self.ps_server = None
        self.lc_connection = None
        self.actor_config = None
        self.critic_config = None
        self.actor_params = None
        self.critic_params = None

    def start_ps_server(self):
        print("Starting Parameter Server")
        self.ps_server.start()

    @classmethod
    def process_starter(cls, config):
        print(f"Parameter Server Process Started: {os.getpid()}")

        signal(SIGINT, ParameterMain.process_terminator)
        signal(SIGTERM, ParameterMain.process_terminator)

        # Starting Parameter Server
        start_event = threading.Event()
        arch_push_event = threading.Event()
        first_param_push_event = threading.Event()
        param_object = ParameterMain()
        ParameterMain.parameter_main_object = param_object
        ps_service = classpartial(ParameterService, start_event, arch_push_event, first_param_push_event, param_object)
        param_object.ps_server = ThreadedServer(ps_service, port=config["param_server_port"], protocol_config={'allow_public_attrs': True,})
        t1 = threading.Thread(target=param_object.start_ps_server)
        t1.start()

        # Sending confirmation to LC about successful process start
        param_object.lc_connection = rpyc.connect("localhost", port=config["lcs_server_port"])
        param_object.lc_connection.root.component_started_confirmation("param")

        # Not sure what to do with the object but still keeping it for potential future use

        # Waiting for start confirmation from LC
        start_event.wait()
        t1.join()

    @classmethod
    def process_terminator(cls, signum, frame):
        param_object = ParameterMain.parameter_main_object
        param_object.close()
        exit(0)

    def close(self):
        self.lc_connection.close()
        self.ps_server.close()


# ======================= Data Accumulator Process =========================================

@rpyc.service
class DataAccumulatorService(rpyc.Service):
    # Accumulates different kinds of data into a thread-safe queue
    def __init__(self, start_event, pusher):
        self.start_event = start_event
        self.pusher = pusher

    @rpyc.exposed
    def start_work(self):
        self.start_event.set()

    @rpyc.exposed
    def push_actor_transition_data(self, data):
        self.pusher.tsqueue.put(data)

    @rpyc.exposed
    def push_actor_log_data(self, data):
        self.pusher.logqueue.put(data)

    @rpyc.exposed
    def collect_accum_data(self, train_step):
        threading.Thread(target=self.pusher.collect_and_push_data, args=(train_step, )).start()


class Pusher:
    # Pushes the data in the queue into algorithm process
    pusher_object = None

    def __init__(self, learner_parameters):
        self.das_server = None
        self.lc_connection = None
        self.alg_connection = None
        self.tsqueue = None
        self.logqueue = None
        self.agent_path = learner_parameters["agent_path"]
        self.actors = {}
        self.log_path = os.path.join(self.agent_path, "actor_logs")
        os.makedirs(self.log_path, exist_ok=True)

    def start_das_server(self):
        print("Starting Data Accumulator Server")
        self.das_server.start()

    @classmethod
    def process_starter(cls, learner_parameters, config):
        print(f"Data Accumulator Process Started: {os.getpid()}")

        signal(SIGINT, Pusher.process_terminator)
        signal(SIGTERM, Pusher.process_terminator)

        # Starting Data Accumulator Server
        start_event = threading.Event()
        pusher = Pusher(learner_parameters)
        Pusher.pusher_object = pusher
        das_service = classpartial(DataAccumulatorService, start_event, pusher)
        pusher.das_server = ThreadedServer(das_service, port=config["accum_server_port"])
        t1 = threading.Thread(target=pusher.start_das_server)
        t1.start()

        # Sending confirmation to LC about successful process start
        pusher.lc_connection = rpyc.connect("localhost", port=config["lcs_server_port"])
        pusher.lc_connection.root.component_started_confirmation("accum")

        # Waiting for start confirmation from LC
        start_event.wait()

        # After confirmation from LC, start pusher
        pusher.start(config)

    @classmethod
    def process_terminator(cls, signum, frame):
        pusher = Pusher.pusher_object
        pusher.close()
        exit(0)

    def start(self, config):
        self.alg_connection = rpyc.connect("localhost", port=config["algo_server_port"])
        if self.tsqueue is None:
            self.tsqueue = Queue()
        if self.logqueue is None:
            self.logqueue = Queue()

        while True:
            print(f"\rTS Size={self.tsqueue.qsize()} Log Size={self.logqueue.qsize()}", end=" "*5)
            time.sleep(0.5)

    def collect_and_push_data(self, train_step):
        size = self.tsqueue.qsize()
        data_list = []
        while size > 0:
            data_list.append(self.tsqueue.get())
            size -= 1
        try:
            self.alg_connection.root.push_replay_data(json.dumps(data_list))
        except:
            pass

        size = self.logqueue.qsize()
        data_list = []
        while size > 0:
            data_list.append(json.loads(self.logqueue.get()))
            size -= 1

        for elem in data_list:
            if elem["actor"] not in self.actors.keys():
                actor_data = {"episode": [], "train_step": []}
                for key in elem["log_data"].keys():
                    actor_data[key] = []
                self.actors[elem["actor"]] = actor_data
            self.actors[elem["actor"]]["train_step"].append(train_step)
            self.actors[elem["actor"]]["episode"].append(elem["episode"])
            for key in elem["log_data"].keys():
                self.actors[elem["actor"]][key].append(elem["log_data"][key])

        for key, values in self.actors.items():
            pd.DataFrame(values).to_csv(os.path.join(self.log_path, f"{key}.csv"), index=False)

    def close(self):
        self.lc_connection.close()
        self.das_server.close()
