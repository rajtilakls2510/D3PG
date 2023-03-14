# from tensorflow.keras import Model
# import tensorflow.keras.layers as layers
#
# inp = layers.Input(shape=(512, ))
# x = layers.Dense(64, activation = "relu")(inp)
# x = layers.Dense(64, activation = "relu")(x)
# x = layers.Dense(64, activation = "relu")(x)
# out = layers.Dense(10, activation = "softmax")(x)
#
# model = Model(inputs = inp, outputs = out)
#
# model.compile(loss = "categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#
# print(model.summary())

# import socket
# print(type(socket.gethostbyname(socket.gethostname())))

import rpyc, os
import threading, multiprocessing

from signal import signal
from signal import SIGINT, SIGTERM


def system_closer(already_signalled = False):
    server.close()
    p1.terminate()
    print("RPyC Server Stopped")
    if not already_signalled:
        os.kill(os.getpid(), SIGINT)

def handle_close(signalnum, frame):
    print(f"Signal: {signalnum}")
    system_closer(True)
    exit(0)



class MyService(rpyc.Service):

    def on_connect(self, conn):
        # code that runs when a connection is created
        # (to init the service, if needed)
        print("Client Connected")

    def on_disconnect(self, conn):
        # code that runs after the connection has already closed
        # (to finalize the service, if needed)
        print("Client Disconnected")

    def exposed_get_answer(self, x, callback):  # this is an exposed method
        callback(x+42)
        return x+42

    def exposed_stop(self):
        system_closer()

    exposed_the_real_answer_though = 43  # an exposed attribute

    def get_question(self):  # while this method is not exposed
        return "what is the airspeed velocity of an unladen swallow?"


def startServer(server):
    print("Starting Server...")
    server.start()
    print("Got here")

def doprocess():
    print(f"Sub: {os.getpid()}")
    while True:
        pass

if __name__ == "__main__":
    from rpyc.utils.server import ThreadedServer
    print(f"Main: {os.getpid()}")
    server = ThreadedServer(MyService, port=18861)
    t1 = threading.Thread(target=startServer, args=[server])
    t1.start()

    signal(SIGINT, handle_close)
    signal(SIGTERM, handle_close)
    print("RpyC Server Started")
    p1 = multiprocessing.Process(target=doprocess)
    p1.start()

    # Proceed to main thread logic
    while True:
        pass


