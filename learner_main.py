import learner


def network_creators():
    return None, None


config = {
    "lcs_server_port": 18861,
    "algo_server_port": 18862,
    "accum_server_port": 18863,
    "param_server_port": 18864
}

if __name__ == "__main__":
    learner_coord = learner.LearnerCoordinator(network_creators, config)
    learner_coord.start()
    print("Learner System Started")

    # You gotta keep working for signals to be received
    while True:
        pass