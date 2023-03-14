import actor
import environment


def env_creator():
    return "Env"

config = {
    "num_actors": 5,
    "lcs_server_host": "localhost",
    "lcs_server_port": 18861,
    "acs_server_port": 18865
}
if __name__ == "__main__":
    actor_coord = actor.ActorCoordinator(env_creator, config)
    actor_coord.start()
    print("Actor System Started")