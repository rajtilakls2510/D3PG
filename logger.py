from time import perf_counter

class Logger:
    # Base class for Logging Callbacks

    def __init__(self):
        self.name = "Log"

    def on_task_begin(self, data=None):
        pass

    def on_task_end(self, data=None):
        pass


class LearnerLogger(Logger):
    # Base class for Loggers for Learner process
    pass


class ActorLogger(Logger):
    # Base class for Loggers for Actor process

    def __init__(self):
        super().__init__()

    def on_episode_begin(self, data=None):
        pass

    def on_episode_step(self, data=None):
        pass

    def on_episode_end(self, data=None):
        pass

    def consume_data(self):
        pass


class EpisodeLengthLogger(ActorLogger):

    def __init__(self):
        super().__init__()
        self.name = "EpisodeLength"
        self.length = 0

    def on_episode_begin(self, data=None):
        self.length = 0

    def on_episode_step(self, data=None):
        self.length += 1

    def consume_data(self):
        return int(self.length)


class TotalRewardLogger(ActorLogger):

    def __init__(self):
        super().__init__()
        self.name = "TotalReward"
        self.reward = 0

    def on_episode_begin(self, data=None):
        self.reward = 0

    def on_episode_step(self, data=None):
        self.reward += data["reward"]

    def consume_data(self):
        return float(self.reward)


class EpisodeTimeLogger(ActorLogger):

    def __init__(self):
        super().__init__()
        self.name = "EpisodeTime"
        self.start_time = 0
        self.end_time = 0

    def on_episode_begin(self, data=None):
        self.start_time = perf_counter()

    def on_episode_end(self, data=None):
        self.end_time = perf_counter()

    def consume_data(self):
        return float(self.end_time - self.start_time)