import threading
import time
import numpy as np
import rpyc
import tensorflow as tf
from rpyc.utils.helpers import classpartial
import multiprocessing
import os
import json
import base64
import cv2
import gc
from signal import signal, SIGINT, SIGTERM
from concurrent.futures import ThreadPoolExecutor


class DDPGActor:
    ddpg_actor_object = None

    def __init__(self, env_creator, config, actor_parameters, i):
        self.accum_server_conn = None
        self.ac_connection = None
        self.bgsrv = None
        self.env_creator = env_creator
        self.mode = actor_parameters["mode"]
        self.std = actor_parameters["std"]
        self.env = env_creator(self.std[i%len(self.std)])
        self.config = config
        self.param_server_conn = None
        self.acs_server_conn = None
        self.actor_network = None
        self.critic_network = None
        self.exploration = actor_parameters["exploration"]
        self.n_fetch = actor_parameters["n_fetch"]
        self.n_push = actor_parameters["n_push"]
        self.transition_buffer = {"current_state": [], "action": [], "reward": [], "next_state": [], "terminated": []}
        self.thread_pool_executor = ThreadPoolExecutor(max_workers=actor_parameters["max_executors"])
        self.show_acting = actor_parameters["show_acting"]
        self.logs = []
        self.actor_num = int(tf.random.uniform(shape=(), maxval=5000, dtype=tf.int32).numpy())
        for l in actor_parameters["logs"]:
            self.logs.append(l())
        print(f"Actor Process Started: {os.getpid()} Actor ID: {self.actor_num} Std: {self.std[i%len(self.std)]}")

    @classmethod
    def process_starter(cls, env_creator, config, actor_parameters, i):
        tf.random.set_seed(tf.random.uniform(shape=(1,), minval=0, maxval=1000, dtype=tf.int32))
        np.random.seed(np.random.randint(0, 1000))
        signal(SIGINT, DDPGActor.process_terminator)
        signal(SIGTERM, DDPGActor.process_terminator)

        actor_object = DDPGActor(env_creator, config, actor_parameters, i)
        DDPGActor.ddpg_actor_object = actor_object
        ac_service = classpartial(ActorClientService, actor_object)
        actor_object.ac_connection = rpyc.connect("localhost", port=config["acs_server_port"], service=ac_service)
        actor_object.bgsrv = rpyc.BgServingThread(actor_object.ac_connection)
        actor_object.act()

    @classmethod
    def process_terminator(cls, signum, frame):
        actor_object = DDPGActor.ddpg_actor_object
        actor_object.close()
        actor_object.ac_connection.close()
        actor_object.bgsrv.stop()
        exit(0)

    def pull_nnet_arch(self):
        actor_config, critic_config = self.param_server_conn.root.get_nnet_arch()
        self.actor_network = tf.keras.Model().from_config(json.loads(actor_config))
        self.critic_network = tf.keras.Model().from_config(json.loads(critic_config))

    def pull_nnet_params(self):
        # Protocol for Receiving:
        # - Convert Json to list of parameters
        # - Base64 Decode Each value of list
        # - Parse Tensor Each value of list
        try:
            actor_weights, critic_weights = self.param_server_conn.root.get_params()
            actor_weights = json.loads(actor_weights)
            critic_weights = json.loads(critic_weights)
            for i in range(len(actor_weights)):
                actor_weights[i] = tf.io.parse_tensor(base64.b64decode(actor_weights[i]), out_type=tf.float32)
            for i in range(len(critic_weights)):
                critic_weights[i] = tf.io.parse_tensor(base64.b64decode(critic_weights[i]), out_type=tf.float32)
            self.actor_network.set_weights(actor_weights)
            self.critic_network.set_weights(critic_weights)
            del actor_weights, critic_weights
            gc.collect()
        except:
            print("\rException: Couldn't pull parameters", end="")

    def push_transition_data(self, transition_buffer):
        try:
            transition_buffer["current_state"] = base64.b64encode(
                tf.io.serialize_tensor(tf.convert_to_tensor(transition_buffer["current_state"])).numpy()).decode(
                "ascii")
            transition_buffer["action"] = base64.b64encode(
                tf.io.serialize_tensor(tf.convert_to_tensor(transition_buffer["action"])).numpy()).decode(
                "ascii")
            transition_buffer["reward"] = base64.b64encode(
                tf.io.serialize_tensor(tf.convert_to_tensor(transition_buffer["reward"])).numpy()).decode(
                "ascii")
            transition_buffer["next_state"] = base64.b64encode(
                tf.io.serialize_tensor(tf.convert_to_tensor(transition_buffer["next_state"])).numpy()).decode(
                "ascii")
            transition_buffer["terminated"] = base64.b64encode(
                tf.io.serialize_tensor(tf.convert_to_tensor(transition_buffer["terminated"])).numpy()).decode(
                "ascii")
            self.accum_server_conn.root.push_actor_transition_data(json.dumps(transition_buffer))
            del transition_buffer
            gc.collect()
        except:
            print("\rException: Couldn't push transition data", end="")

    def push_log_data(self, data):
        try:
            self.accum_server_conn.root.push_actor_log_data(json.dumps(data))
            del data
            gc.collect()
        except:
            print("\r Exception: Couldn't push log data", end="")

    def act(self):
        # Setup Necessary connections to Parameter Server and Data Accumulator Server
        self.param_server_conn = rpyc.connect(self.config["param_server_host"], port=self.config["param_server_port"])
        self.accum_server_conn = rpyc.connect(self.config["accum_server_host"], port=self.config["accum_server_port"])

        # Pulling Neural Networks from Parameter Server
        self.pull_nnet_arch()
        self.pull_nnet_params()

        # Start Running Episodes

        for log in self.logs:
            log.on_task_begin()

        step = 0
        episode = 0
        while True:
            self.env.reset()

            for log in self.logs:
                log.on_episode_begin({"episode": episode})

            current_state, _, _ = self.env.observe()
            current_state = tf.convert_to_tensor(current_state, tf.float32)
            while not self.env.is_episode_finished():
                action, action_value, explored = self.get_action(current_state, explore=self.exploration)
                self.env.take_action(action.numpy())
                next_state, reward, frame = self.env.observe()
                next_state = tf.convert_to_tensor(next_state, tf.float32)
                reward = tf.convert_to_tensor(reward, dtype=tf.float32)

                if self.mode == "train":
                    self.transition_buffer["current_state"].append(current_state)
                    self.transition_buffer["action"].append(action)
                    self.transition_buffer["reward"].append(reward)
                    self.transition_buffer["next_state"].append(next_state)
                    self.transition_buffer["terminated"].append(tf.convert_to_tensor(self.env.is_episode_finished()))

                current_state = next_state

                step_data = {
                    "current_state": current_state.numpy(),
                    "action_value": action_value.numpy(),
                    "action": action.numpy(),
                    "reward": reward.numpy(),
                    "next_state": next_state.numpy(),
                    "explored": explored.numpy(),
                    "frame": frame
                }

                for log in self.logs:
                    log.on_episode_step(step_data)

                if self.show_acting:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    # fontScale
                    fontScale = 0.3

                    # Blue color in BGR
                    color = (255, 0, 0)

                    # Line thickness of 2 px
                    thickness = 2
                    frame = cv2.putText(frame, f"Action: {action}", org=(10,12), fontFace=font, fontScale=fontScale, color=color,lineType=cv2.LINE_AA)
                    frame = cv2.putText(frame, f"Value: {action_value}", org=(10,24), fontFace=font, fontScale=fontScale, color=color,lineType=cv2.LINE_AA)
                    cv2.imshow("Actor: "+str(self.actor_num), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    cv2.waitKey(1000 // 60)
                step += 1

                if step % self.n_fetch == 0:
                    self.thread_pool_executor.submit(self.pull_nnet_params)
                if step % self.n_push == 0:
                    if self.mode == "train":
                        self.thread_pool_executor.submit(self.push_transition_data, self.transition_buffer)
                        self.transition_buffer = {"current_state": [], "action": [], "reward": [], "next_state": [],
                                              "terminated": []}
            for log in self.logs:
                log.on_episode_end()

            data = {"actor": self.actor_num, "episode": episode, "log_data": {}}
            for log in self.logs:
                data["log_data"][log.name] = log.consume_data()
            self.thread_pool_executor.submit(self.push_log_data, data)
            gc.collect()
            episode += 1

        for log in self.logs:
            log.on_task_end()

    def get_action(self, state, explore=0.0):
        state = tf.expand_dims(state, axis=0)
        action = self.actor_network(state)
        explored = tf.constant(False)
        if np.random.uniform(0,1) < explore:
            action = action + tf.convert_to_tensor(self.env.get_random_action(), tf.float32)
            explored = tf.constant(True)
        value = self.critic_network([state, action])
        return action[0], value[0][0], explored

    def get_values(self, states):
        return self.critic_network([states, self.actor_network(states)])

    def close(self):
        # Release any resources here
        self.env.close()
        if self.param_server_conn is not None:
            self.param_server_conn.close()
            self.param_server_conn = None
        self.thread_pool_executor.shutdown()
        if self.show_acting:
            cv2.destroyWindow("Actor: "+str(self.actor_num))


@rpyc.service
class ActorClientService(rpyc.Service):
    # Actor client service to answer coordinator requests

    def __init__(self, actor):
        self.actor = actor


# ========================= Actor Coordinator Process ===============================


@rpyc.service
class ActorCoordinatorService(rpyc.Service):
    # The actor coordinator service
    def __init__(self, start_event, connection_holders):
        self.start_event = start_event
        self.connection_holders = connection_holders

    def on_connect(self, conn):
        self.connection_holders["actors"].append(conn)

    def on_disconnect(self, conn):
        self.connection_holders["actors"].remove(conn)

    @rpyc.exposed
    def start_work(self):
        self.start_event.set()

    @rpyc.exposed
    def stop(self):
        ActorCoordinator.process_terminator(None, None)


class ActorCoordinator:
    # Main Actor process is started with the object of this class to start the actor system
    reference_holders = {"actor_processes": []}
    connection_holders = {"actors": []}
    lc_connection = None
    acs_server = None

    def __init__(self, env_creator, config, actor_parameters):
        self.env_creator = env_creator
        self.config = config
        self.actor_parameters = actor_parameters

    def start_acs_server(self):
        print("Starting Actor Coordinator Server")
        ActorCoordinator.acs_server.start()

    def start(self):

        signal(SIGINT, ActorCoordinator.process_terminator)
        signal(SIGTERM, ActorCoordinator.process_terminator)

        # Starting Actor Coordinator Server
        start_event = threading.Event()
        acs_service = classpartial(ActorCoordinatorService, start_event, ActorCoordinator.connection_holders)
        ActorCoordinator.acs_server = rpyc.ThreadedServer(acs_service, port=self.config["acs_server_port"])
        t1 = threading.Thread(target=self.start_acs_server)
        t1.start()

        # Sending Confirmation to LC about successful process start
        ActorCoordinator.lc_connection = rpyc.connect(self.config["lcs_server_host"],
                                                      port=self.config["lcs_server_port"])
        ActorCoordinator.lc_connection.root.component_started_confirmation("actors",
                                                                           info={"host": self.config["acs_server_host"],
                                                                                 "port": self.config[
                                                                                     "acs_server_port"]})

        # Waiting for start confirmation from LC
        start_event.wait()

        # Starting Actors after confirmation from LC
        print("Starting Actor Processes")
        for i in range(self.config["num_actors"]):
            p = multiprocessing.Process(target=DDPGActor.process_starter,
                                        args=(self.env_creator, self.config, self.actor_parameters, i))
            p.start()
            ActorCoordinator.reference_holders["actor_processes"].append(p)

    def monitor_system(self):
        # Monitors the system. Makes sure all child processes are up and running. It is called after start is
        # called
        # You gotta keep working for signals to be received
        while True:
            time.sleep(1)

    @classmethod
    def process_terminator(cls, signum, frame):
        print("Terminating Actor System")

        for actor_process in ActorCoordinator.reference_holders["actor_processes"]:
            actor_process.terminate()

        for actor_process in ActorCoordinator.reference_holders["actor_processes"]:
            actor_process.join()

        ActorCoordinator.lc_connection.close()
        ActorCoordinator.acs_server.close()
        if signum is None:
            os.kill(os.getpid(), SIGTERM)
        exit(0)
