import actor

config = {
    "lcs_server_host": "localhost",
    "lcs_server_port": 18861,
    "acs_server_port": 18867
}
if __name__ == "__main__":
    actor_coord = actor.ActorCoordinator(config)
    actor_coord.start()
    print("Actor System Started")