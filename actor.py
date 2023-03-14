import threading
import rpyc
import tensorflow as tf
from rpyc.utils.helpers import classpartial
import socket
import multiprocessing
import os


class DDPGActor:
    ac_connection = None
    ddpg_actor_object = None

    def __init__(self, env):
        self.env = env
        self.actor_network = None
        self.critic_network = None

    @classmethod
    def process_starter(cls, env_creator, config):
        print(f"Actor Process Started: {os.getpid()}")

        DDPGActor.ac_connection = rpyc.connect("localhost", port=config["acs_server_port"], service=ActorClientService)
        DDPGActor.ddpg_actor_object = DDPGActor(env_creator())
        DDPGActor.ddpg_actor_object.act()

    def act(self):
        # TODO: Write acting logic
        pass

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


@rpyc.service
class ActorClientService(rpyc.Service):
    # Actor client service to answer coordinator requests
    pass

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


class ActorCoordinator:
    # Main Actor process is started with the object of this class to start the actor system
    def __init__(self, env_creator, config):
        self.env_creator = env_creator
        self.reference_holders = {"actor_processes": []}
        self.connection_holders = {"actors": []}
        self.config = config
        self.acs_server = None

    def start_acs_server(self):
        print("Starting Actor Coordinator Server")
        self.acs_server.start()

    def start(self):

        # Starting Actor Coordinator Server
        start_event = threading.Event()
        acs_service = classpartial(ActorCoordinatorService, start_event, self.connection_holders)
        self.acs_server = rpyc.ThreadedServer(acs_service, port=self.config["acs_server_port"])
        t1 = threading.Thread(target=self.start_acs_server)
        t1.start()

        # Sending Confirmation to LC about successful process start
        lc_connection = rpyc.connect(self.config["lcs_server_host"], port=self.config["lcs_server_port"])
        lc_connection.root.component_started_confirmation("actors", info={"host": socket.gethostbyname(socket.gethostname()), "port": self.config["acs_server_port"]})

        # Waiting for start confirmation from LC
        start_event.wait()

        # Starting Actors after confirmation from LC
        print("Starting Actor Processes")
        for i in range(self.config["num_actors"]):
            p = multiprocessing.Process(target=DDPGActor.process_starter, args=(self.env_creator, self.config,))
            p.start()
            self.reference_holders["actor_processes"].append(p)




    def monitor_system(self):
        # Monitors the system. Makes sure all child processes are up and running. It is called after start is
        # called
        pass