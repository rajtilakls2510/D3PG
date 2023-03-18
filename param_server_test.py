import rpyc
import tensorflow as tf
import json
import base64

# def rpyc_deep_copy(obj):
#     """
#     Makes a deep copy of netref objects that come as a result of RPyC remote method calls.
#     When RPyC client obtains a result from the remote method call, this result may contain
#     non-scalar types (List, Dict, ...) which are given as a wrapper class (a netref object).
#     This class does not have all the standard attributes (e.g. dict.tems() does not work)
#     and in addition the objects only exist while the connection is active (are weekly referenced).
#     To have a retuned value represented by python's native datatypes and to by able to use it
#     after the connection is terminated, this routine makes a recursive copy of the given object.
#     Currently, only `list` and `dist` types are supported for deep_copy, but other types may be
#     added easily.
#     Note there is allow_attribute_public option for RPyC connection, which may solve the problem too,
#     but it have not worked for me.
#     Example:
#         s = rpyc.connect(host1, port)
#         result = rpyc_deep_copy(s.root.remote_method())
#         # if result is a Dict:
#         for k,v in result.items(): print(k,v)
#     """
#     if (isinstance(obj, list)):
#         copied_list = []
#         for value in obj: copied_list.append(rpyc_deep_copy(value))
#         return copied_list
#     elif (isinstance(obj, dict)):
#         copied_dict = {}
#         for key in obj: copied_dict[key] = rpyc_deep_copy(obj[key])
#         return copied_dict
#     else:
#         return obj

# Set memory_growth option to True otherwise tensorflow will eat up all GPU memory
try:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

conn = rpyc.connect("localhost", port=18864)

actor_config, critic_config = conn.root.get_nnet_arch()
actor_network = tf.keras.Model().from_config(json.loads(actor_config))
critic_network = tf.keras.Model().from_config(json.loads(critic_config))

print(actor_network.summary())
print(critic_network.summary())
actor_weights, critic_weights = conn.root.get_params()

# Protocol for Receiving:
# - Convert Json to list of parameters
# - Base64 Decode it
# - Parse Tensor

actor_weights = json.loads(actor_weights)
critic_weights = json.loads(critic_weights)
for i in range(len(actor_weights)):
    actor_weights[i] = tf.io.parse_tensor(base64.b64decode(actor_weights[i]), out_type=tf.float32)
for i in range(len(critic_weights)):
    critic_weights[i] = tf.io.parse_tensor(base64.b64decode(critic_weights[i]), out_type=tf.float32)
actor_network.set_weights(actor_weights)
critic_network.set_weights(critic_weights)
print(actor_network.get_weights())
# print(critic_network.get_weights())
