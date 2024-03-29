import threading
import tensorflow as tf
import os


class ReplayBuffer:
    # Base class for any kind of ReplayBuffer
    # Description: This class is used to store experience replays by different kinds of
    #           RL Algorithms.

    def __init__(self, max_transitions=1000, continuous_actions=False):
        self.max_transitions = max_transitions
        self.continuous_actions = continuous_actions

    # Inserts transition to buffer
    def insert_transition(self, transition):
        pass

    # Inserts a batch of transitions to the buffer
    def insert_batch_transitions(self, transitions):
        pass

    # Samples a batch of transitions from the buffer
    def sample_batch_transitions(self, batch_size=16):
        pass

    # Returns the number of transitions in the replay buffer
    def size(self):
        pass

    # Saves the buffer
    def save(self, path=""):
        pass

    # Loads the buffer
    def load(self, path=""):
        pass



class UniformReplay(ReplayBuffer):
    # This class is an implementation of a simple Uniform Replay Buffer

    def __init__(self, max_transitions=1000, continuous_actions=False):
        super().__init__(max_transitions, continuous_actions)
        self.current_states = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        if self.continuous_actions:
            self.actions = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        else:
            self.actions = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False)
        self.rewards = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        self.next_states = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        self.terminals = tf.TensorArray(tf.bool, size=0, dynamic_size=True, clear_after_read=False)
        self.current_index = 0
        self.lock = threading.Lock()

    def _insert_transition_at(self, transition, index):
        self.current_states = self.current_states.write(index, transition[0])
        self.actions = self.actions.write(index, transition[1])
        self.rewards = self.rewards.write(index, transition[2])
        self.next_states = self.next_states.write(index, transition[3])
        self.terminals = self.terminals.write(index, transition[4])

    # Transition Format: current_state, action, reward, next_state, terminal step or not
    def insert_transition(self, transition):
        self._insert_transition_at(transition, self.current_index % self.max_transitions)
        self.current_index += 1

    def insert_batch_transitions(self, transitions):
        batch_size = transitions["current_state"].shape[0]
        self.lock.acquire()
        indices = [(self.current_index+i) % self.max_transitions for i in range(batch_size)]
        self.current_states = self.current_states.scatter(indices, transitions["current_state"])
        self.actions = self.actions.scatter(indices, transitions["action"])
        self.rewards = self.rewards.scatter(indices, transitions["reward"])
        self.next_states = self.next_states.scatter(indices, transitions["next_state"])
        self.terminals = self.terminals.scatter(indices, transitions["terminated"])
        self.current_index += batch_size
        self.lock.release()

    def sample_batch_transitions(self, batch_size=16):
        self.lock.acquire()
        buf_len = self.current_states.size()
        if buf_len <= batch_size:
            sampled_indices = tf.random.uniform(shape=(buf_len,), maxval=buf_len, dtype=tf.int32)
        else:
            sampled_indices = tf.random.uniform(shape=(batch_size,), maxval=buf_len, dtype=tf.int32)

        current_states, actions, rewards, next_states, terminals = self.current_states.gather(sampled_indices), self.actions.gather(sampled_indices), self.rewards.gather(
            sampled_indices), self.next_states.gather(sampled_indices), self.terminals.gather(sampled_indices)
        self.lock.release()
        return current_states, actions, rewards, next_states, terminals

    def size(self):
        return self.current_states.size()

    def save(self, path=""):
        self.lock.acquire()
        tf.io.write_file(os.path.join(path, "current_states.tfw"), tf.io.serialize_tensor(self.current_states.stack()))
        tf.io.write_file(os.path.join(path, "actions.tfw"), tf.io.serialize_tensor(self.actions.stack()))
        tf.io.write_file(os.path.join(path, "rewards.tfw"), tf.io.serialize_tensor(self.rewards.stack()))
        tf.io.write_file(os.path.join(path, "next_states.tfw"), tf.io.serialize_tensor(self.next_states.stack()))
        tf.io.write_file(os.path.join(path, "terminals.tfw"), tf.io.serialize_tensor(self.terminals.stack()))
        self.lock.release()

    def load(self, path=""):
        self.lock.acquire()
        try:
            current_states = self.current_states.unstack(
                tf.io.parse_tensor(tf.io.read_file(os.path.join(path, "current_states.tfw")), tf.float32))

            if self.continuous_actions:
                actions = self.actions.unstack(
                    tf.io.parse_tensor(tf.io.read_file(os.path.join(path, "actions.tfw")), tf.float32))
            else:
                actions = self.actions.unstack(
                    tf.io.parse_tensor(tf.io.read_file(os.path.join(path, "actions.tfw")), tf.int32))
            rewards = self.rewards.unstack(
                tf.io.parse_tensor(tf.io.read_file(os.path.join(path, "rewards.tfw")), tf.float32))
            next_states = self.next_states.unstack(
                tf.io.parse_tensor(tf.io.read_file(os.path.join(path, "next_states.tfw")), tf.float32))
            terminals = self.terminals.unstack(
                tf.io.parse_tensor(tf.io.read_file(os.path.join(path, "terminals.tfw")), tf.bool))
            self.current_states = current_states
            self.actions = actions
            self.rewards = rewards
            self.next_states = next_states
            self.terminals = terminals
            self.current_index = self.current_states.size().numpy()
            print("Found", self.current_states.size().numpy(), "transitions")
        except Exception as e:
            print("No Experience Replay found")
        self.lock.release()